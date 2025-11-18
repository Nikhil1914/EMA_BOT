import datetime as dt
import pandas as pd
import numpy as np
import threading
import time
from websocket import WebSocketApp

from config import MARKET_OPEN, MARKET_CLOSE, INTRADAY_SQUAREOFF, MAX_CANDLE_STORAGE, WEBSOCKET_RECONNECT_DELAY
from fyers_helper import get_ltp, place_market_order, close_position, map_live_symbol, ACCESS_TOKEN, CLIENT_ID


# =================================================================
# GLOBAL STATE
# =================================================================
LIVE_STATE = {
    "running": False,
    "ws": None,
    "current_symbol": None,
    "tf": "1",
    "candle_buffer": [],
    "position": "flat",
    "entry_price": None,
    "entry_time": None,
    "tp_price": None,
    "sl_price": None,
    "last_signal": None,
    "trade_mode": "Intraday",
    "trade_side": "Long Only",
    "qty": 1,
    "product": "INTRADAY",
    "ma_type": "SMA",
    "fast": 9,
    "slow": 21,
    "tp_type": "Points",
    "tp_value": 100,
    "sl_type": "Points",
    "sl_value": 50,
    "log": [],
    "lock": threading.Lock()
}


# =================================================================
# LOGGING
# =================================================================
def log_msg(msg):
    with LIVE_STATE["lock"]:
        LIVE_STATE["log"].append(f"{dt.datetime.now()} | {msg}")
    print(msg)


# =================================================================
# STRATEGY MODULE
# =================================================================
def calc_tp_sl(entry, tp_type, tp_value, sl_type, sl_value, direction):
    if tp_type == "Points":
        tp = entry + tp_value if direction == "long" else entry - tp_value
    else:
        tp = entry * (1 + tp_value / 100) if direction == "long" else entry * (1 - tp_value / 100)

    if sl_type == "Points":
        sl = entry - sl_value if direction == "long" else entry + sl_value
    else:
        sl = entry * (1 - sl_value / 100) if direction == "long" else entry * (1 + sl_value / 100)

    return tp, sl


def calculate_ma(df, ma_type, fast, slow):
    if ma_type == "SMA":
        df["fast_ma"] = df["close"].rolling(fast).mean()
        df["slow_ma"] = df["close"].rolling(slow).mean()
    else:
        df["fast_ma"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["slow_ma"] = df["close"].ewm(span=slow, adjust=False).mean()

    df["signal"] = 0
    df.loc[(df["fast_ma"] > df["slow_ma"]) & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1)), "signal"] = 1
    df.loc[(df["fast_ma"] < df["slow_ma"]) & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1)), "signal"] = -1

    return df


def get_signal_from_candles():
    df = pd.DataFrame(LIVE_STATE["candle_buffer"], columns=["time", "open", "high", "low", "close"])
    if len(df) < LIVE_STATE["slow"] + 2:
        return None

    df = calculate_ma(df, LIVE_STATE["ma_type"], LIVE_STATE["fast"], LIVE_STATE["slow"])
    last = df.iloc[-1]
    if last["signal"] == 1:
        return "long"
    if last["signal"] == -1:
        return "short"
    return None


# =================================================================
# ORDER EXECUTION
# =================================================================
def enter_position(direction):
    price = get_ltp(LIVE_STATE["current_symbol"])
    if not price:
        return

    LIVE_STATE["position"] = direction
    LIVE_STATE["entry_price"] = price
    LIVE_STATE["entry_time"] = dt.datetime.now()
    LIVE_STATE["tp_price"], LIVE_STATE["sl_price"] = calc_tp_sl(
        price, LIVE_STATE["tp_type"], LIVE_STATE["tp_value"], LIVE_STATE["sl_type"], LIVE_STATE["sl_value"], direction
    )

    side = "BUY" if direction == "long" else "SELL"
    resp = place_market_order(LIVE_STATE["current_symbol"], side, LIVE_STATE["qty"], LIVE_STATE["product"])
    log_msg(f"{direction.upper()} ENTRY @ {price} | {resp}")


def exit_position(reason):
    direction = LIVE_STATE["position"]
    resp = close_position(LIVE_STATE["current_symbol"], direction, LIVE_STATE["qty"], LIVE_STATE["product"])
    log_msg(f"EXIT {direction.upper()} | {reason} | {resp}")

    LIVE_STATE["position"] = "flat"
    LIVE_STATE["entry_price"] = None
    LIVE_STATE["tp_price"] = None
    LIVE_STATE["sl_price"] = None


# =================================================================
# STRATEGY ENGINE
# =================================================================
def evaluate_strategy(latest_price):
    now = dt.datetime.now().time()
    market_open = dt.time(*MARKET_OPEN)
    market_close = dt.time(*MARKET_CLOSE)
    sqoff_time = dt.time(*INTRADAY_SQUAREOFF)

    # Market hours restriction
    if not (market_open <= now <= market_close):
        return

    # Intraday squareoff
    if LIVE_STATE["trade_mode"] == "Intraday" and now >= sqoff_time:
        if LIVE_STATE["position"] != "flat":
            exit_position("Intraday EOD square-off")
        return

    # TP / SL
    if LIVE_STATE["position"] != "flat":
        if LIVE_STATE["position"] == "long":
            if latest_price >= LIVE_STATE["tp_price"]:
                exit_position("TP HIT")
            elif latest_price <= LIVE_STATE["sl_price"]:
                exit_position("SL HIT")
        else:
            if latest_price <= LIVE_STATE["tp_price"]:
                exit_position("TP HIT")
            elif latest_price >= LIVE_STATE["sl_price"]:
                exit_position("SL HIT")

    signal = get_signal_from_candles()
    if signal is None:
        return

    # Trade side rules
    if LIVE_STATE["trade_side"] == "Long Only" and signal == "short":
        return
    if LIVE_STATE["trade_side"] == "Short Only" and signal == "long":
        return

    # Flip
    if LIVE_STATE["position"] == "long" and signal == "short":
        exit_position("FLIP")
        enter_position("short")
    elif LIVE_STATE["position"] == "short" and signal == "long":
        exit_position("FLIP")
        enter_position("long")
    elif LIVE_STATE["position"] == "flat":
        enter_position(signal)

    LIVE_STATE["last_signal"] = signal


# =================================================================
# CANDLE BUILDER
# =================================================================
def aggregate_tick(price):
    ts = dt.datetime.now()
    tf = int(LIVE_STATE["tf"])

    if not LIVE_STATE["candle_buffer"]:
        LIVE_STATE["candle_buffer"].append([ts, price, price, price, price])
        return

    last = LIVE_STATE["candle_buffer"][-1]
    diff = (ts - last[0]).total_seconds() / 60.0

    if diff >= tf:
        LIVE_STATE["candle_buffer"].append([ts, price, price, price, price])
        if len(LIVE_STATE["candle_buffer"]) > MAX_CANDLE_STORAGE:
            LIVE_STATE["candle_buffer"] = LIVE_STATE["candle_buffer"][-MAX_CANDLE_STORAGE:]
    else:
        last[2] = max(last[2], price)
        last[3] = min(last[3], price)
        last[4] = price


# =================================================================
# WEBSOCKET HANDLERS
# =================================================================
def on_message(ws, msg):
    try:
        data = eval(msg)
        if "ltp" not in data:
            return
        price = float(data["ltp"])
    except:
        return

    aggregate_tick(price)
    evaluate_strategy(price)


def on_open(ws):
    log_msg("WebSocket connected")

    payload = {"symbol": [LIVE_STATE["current_symbol"]], "dataType": "symbolUpdate"}
    ws.send(str(payload).replace("'", '"'))

    log_msg(f"Subscribed to {LIVE_STATE['current_symbol']}")


def on_close(ws, *args):
    log_msg("WebSocket closed")
    if LIVE_STATE["running"]:
        time.sleep(WEBSOCKET_RECONNECT_DELAY)
        start_websocket()


def on_error(ws, error):
    log_msg(f"WebSocket ERROR: {error}")


def start_websocket():
    access = f"{CLIENT_ID}:{ACCESS_TOKEN}"
    ws_url = "wss://api.fyers.in/socket/v3/data"

    ws = WebSocketApp(
        ws_url,
        header=[f"access_token:{access}"],
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )

    LIVE_STATE["ws"] = ws
    threading.Thread(target=ws.run_forever, daemon=True).start()


# =================================================================
# PUBLIC API USED BY STREAMLIT
# =================================================================
def start_live_engine(symbol, timeframe):
    LIVE_STATE["current_symbol"] = map_live_symbol(symbol)
    LIVE_STATE["tf"] = timeframe
    LIVE_STATE["running"] = True

    log_msg(f"Starting LIVE engine for {LIVE_STATE['current_symbol']} | TF={timeframe}")
    start_websocket()


def stop_live_engine():
    LIVE_STATE["running"] = False
    if LIVE_STATE["ws"]:
        LIVE_STATE["ws"].close()
    log_msg("Stopped LIVE engine")
