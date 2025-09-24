# main.py
"""
Final Scalping Signal Bot (Futures, ccxt) — per user spec, with requested improvements:
- Adaptive Bollinger filter (sideways vs trending) on 15m
- PnL calculation uses integer contracts (floor)
- Monitor loop tightened for Partial TP / BreakEven confirmation (price & small time buffer)
- Symbol-wide cooldown on close: both 5m & 15m blocked for 15 minutes
- Retrain threshold increased to 100; BreakEven counts as loss (0)
- Risk:Reward enforced to 1:2 (TP distance = 2 * SL distance), while keeping volatility scaling
"""

import os, time, json, threading, traceback, math, pickle
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import ccxt  # <-- Futures via ccxt

# ---------------- Config ----------------
# NOTE: Futures symbol format in ccxt for MEXC USDT-M swaps: "ETH/USDT:USDT", "SOL/USDT:USDT"
SYMBOLS_MANDATORY = ["ETH/USDT:USDT", "SOL/USDT:USDT"]  # force-watch these
DYNAMIC_TOP_N = 18

CHECK_INTERVAL_MAIN = 5       # main loop check interval (seconds)
CHECK_INTERVAL_MONITOR = 5    # monitor loop check interval (seconds)

MIN_CONF_POST = 0.70
HIGH_CONF = 0.85
PARTIAL_TP_RATIO = 0.5
UTBOT_ATR_PERIOD = 10
UTBOT_SENS = 1
LRC_SMOOTHING = 11
LRC_LENGTH = 11
VOLUME_LOOKBACK = 20

MODEL_FILE = "model.pkl"

# ---- TP/SL multipliers per timeframe ---- (kept as base multipliers; final R:R enforced 1:2)
TP_MULT_5M = 1.8
SL_MULT_5M = 1.2
TP_MULT_15M = 3.2
SL_MULT_15M = 2.2

# ---------- Indicator settings (run ONLY on 15m per spec) ----------
ST_PERIOD = 10
ST_MULTIPLIER = 3.0
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
MIN_24H_VOL_USDT = 10_000

# Bollinger settings
BB_PERIOD = 20
BB_STD = 2.0

# Combined strength weights (UT, LRC)
W_UT = 0.6
W_LRC = 0.4

# Cooldown (used previously for candles) - we will set symbol-wide cooldown on close
COOLDOWN_CANDLES = 2
COOLDOWN_MINUTES_AFTER_CLOSE = 15  # NEW: symbol-wide cooldown minutes after any close

# Margin default (per-trade, cross assumed)
MARGIN_USD = 20.0
DEFAULT_LEVERAGE = 10  # used only as fallback if max not found

# Retrain threshold (increased per request)
N_TRADES_FOR_RETRAIN = 100

# Secrets
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MEXC_API_KEY = os.getenv("MEXC_API_KEY")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY")
if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MEXC_API_KEY, MEXC_SECRET_KEY]):
    raise SystemExit("Missing secrets: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, MEXC_API_KEY, MEXC_SECRET_KEY")

# ---------------- ccxt Futures Exchange Init ----------------
exchange = ccxt.mexc({
    "apiKey": MEXC_API_KEY,
    "secret": MEXC_SECRET_KEY,
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap",  # USDT-M futures
    }
})

# ---------------- Persistence ----------------
USE_REPLIT_DB = False
try:
    from replit import db
    USE_REPLIT_DB = True
except Exception:
    USE_REPLIT_DB = False

DB_FILE = "replit_db_fallback.json"

def _read_json():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json(d):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass

def db_get(k, default=None):
    if USE_REPLIT_DB:
        return db.get(k, default)
    d = _read_json()
    return d.get(k, default)

def db_set(k, v):
    if USE_REPLIT_DB:
        db[k] = v; return
    d = _read_json()
    d[k] = v
    _write_json(d)

def db_push(k, item):
    arr = db_get(k, [])
    arr.append(item)
    db_set(k, arr)

if db_get("accuracy_stats") is None:
    db_set("accuracy_stats", {
        "AI+INDICATORS":{"wins":0,"losses":0,"trades":0},
        "AI ONLY":{"wins":0,"losses":0,"trades":0}
    })
if db_get("retrain_counter") is None:
    db_set("retrain_counter", 0)
if db_get("cooldowns") is None:
    db_set("cooldowns", {})  # { "SYMBOL|TF": cooldown_until_timestamp_ms }

# ---------------- Helpers ----------------
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=8)
        r.raise_for_status()
    except Exception as e:
        print("Telegram send failed:", e)

def safe_print_exc(tag="ERROR"):
    print(f"[{tag}] {datetime.utcnow().isoformat()}")
    traceback.print_exc()
    try:
        send_telegram(f"⚠ Bot Error: {tag}\nSee console logs.")
    except:
        pass

def format_price(p: float) -> str:
    try:
        if p is None or p == 0: return "0"
        s = f"{p:.8f}" if abs(p) < 1 else f"{p:.6f}"
        s = s.rstrip('0').rstrip('.')
        return s
    except:
        return str(p)

# ---------------- Futures (ccxt) helpers ----------------
def fetch_markets_once() -> Dict[str, Any]:
    try:
        return exchange.load_markets()
    except Exception:
        safe_print_exc("load_markets")
        return {}

def _is_usdt_swap_sym(market: Dict[str, Any]) -> bool:
    try:
        return market.get("type") == "swap" and market.get("linear") and market.get("quote") == "USDT"
    except:
        return False

def fetch_top_n_symbols(n=500) -> List[str]:
    """Pick top-N USDT-M swap symbols by volume or OI (best-effort)."""
    try:
        markets = fetch_markets_once()
        candidates = []
        for sym, m in markets.items():
            if _is_usdt_swap_sym(m):
                vol = m.get("info", {}).get("volume24h", None)
                if vol is None:
                    vol = m.get("info", {}).get("turnover24h", 0)
                try:
                    vol = float(vol) if vol is not None else 0.0
                except:
                    vol = 0.0
                candidates.append((sym, vol))
        # sort desc by volume proxy
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [s for s,_ in candidates[:n]]
    except Exception:
        safe_print_exc("fetch_top_n_symbols")
        return []

def fetch_klines(symbol: str, interval="15m", limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
        # ccxt returns: [ts, open, high, low, close, volume]
        return [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in ohlcv] if ohlcv else []
    except Exception:
        return []

def fetch_ticker(symbol: str) -> Dict[str, Any]:
    try:
        t = exchange.fetch_ticker(symbol)
        return t or {}
    except Exception:
        return {}

def fetch_orderbook(symbol: str, limit=50) -> Dict[str, Any]:
    try:
        return exchange.fetch_order_book(symbol, limit=limit) or {"bids": [], "asks": []}
    except Exception:
        return {"bids": [], "asks": []}

def fetch_last_price(symbol: str) -> Optional[float]:
    try:
        t = fetch_ticker(symbol)
        info = t.get("info", {}) if isinstance(t, dict) else {}
        mark = None
        for k in ("markPrice", "lastPrice", "last"):
            v = info.get(k)
            if v is not None:
                try:
                    mark = float(v); break
                except:
                    pass
        if mark is not None:
            return mark
        last = t.get("last")
        return float(last) if last is not None else None
    except Exception:
        return None

# Attempt to fetch & set leverage to max for a given symbol (best-effort). Cache result.
_LEVERAGE_CACHE: Dict[str, int] = {}
def fetch_and_set_max_leverage(symbol: str) -> int:
    if symbol in _LEVERAGE_CACHE:
        return _LEVERAGE_CACHE[symbol]
    try:
        markets = fetch_markets_once()
        m = markets.get(symbol, {})
        limits = m.get("limits", {}).get("leverage", {})
        max_lev = limits.get("max") or DEFAULT_LEVERAGE
        # set cross leverage to max
        try:
            exchange.set_leverage(int(max_lev), symbol, params={"marginMode": "cross"})
        except Exception:
            # fallback without params for exchanges that don't require marginMode
            try:
                exchange.set_leverage(int(max_lev), symbol)
            except Exception:
                pass
        _LEVERAGE_CACHE[symbol] = int(max_lev)
        return int(max_lev)
    except Exception:
        _LEVERAGE_CACHE[symbol] = DEFAULT_LEVERAGE
        return DEFAULT_LEVERAGE

# ---------------- Indicators ----------------
def atr(khist: List[List[Any]], period=10):
    trs=[]
    try:
        for i in range(1, min(len(khist), period+1)):
            high = float(khist[-i][2]); low = float(khist[-i][3])
            prev_close = float(khist[-i-1][4] if len(khist) > i else khist[-i][4])
            trs.append(max(high-low, abs(high-prev_close), abs(low-prev_close)))
        return sum(trs)/max(1,len(trs))
    except Exception:
        return 0.0

def utbot_flags(khist: List[List[Any]]) -> Tuple[Optional[str], float]:
    try:
        if len(khist) < UTBOT_ATR_PERIOD+2: return None, 0.0
        xatr = atr(khist, UTBOT_ATR_PERIOD)
        a = UTBOT_SENS
        src = float(khist[-1][4]); prev = float(khist[-2][4])
        stop = src - a * xatr if src>=prev else src + a * xatr
        if src > stop and src > prev:
            return "BUY", 0.6 + min(0.39, (src-prev)/max(1e-8, xatr)*0.1)
        if src < stop and src < prev:
            return "SELL", 0.6 + min(0.39, (prev-src)/max(1e-8, xatr)*0.1)
        return "NEUTRAL", 0.1
    except:
        return None, 0.0

def linreg_flags(khist: List[List[Any]]) -> Tuple[Optional[str], float]:
    try:
        n = LRC_LENGTH
        if len(khist) < n: return None, 0.0
        closes = [float(k[4]) for k in khist[-n:]]
        slope = closes[-1] - closes[0]
        return ("BUY", abs(slope)) if slope>0 else ("SELL", abs(slope))
    except:
        return None, 0.0

def volume_level(khist: List[List[Any]])->str:
    try:
        vols = [float(k[5]) for k in khist[-(VOLUME_LOOKBACK+1):]]
        if len(vols) < 2: return "Unknown"
        last = vols[-1]; avg = sum(vols[:-1])/max(1,len(vols[:-1]))
        return "High" if last >= 1.2*avg else "Low"
    except:
        return "Unknown"

def ema_from_list(values: List[float], period: int) -> List[float]:
    if not values or period <= 0:
        return []
    emas = []
    k = 2 / (period + 1)
    ema_prev = sum(values[:period]) / period if len(values) >= period else values[0]
    for i, v in enumerate(values):
        if i == 0:
            ema_prev = v if len(values) < period else ema_prev
            emas.append(ema_prev)
        else:
            ema_prev = (v - ema_prev) * k + ema_prev
            emas.append(ema_prev)
    return emas

def ema_crossover(khist: List[List[Any]], fast=EMA_FAST, slow=EMA_SLOW) -> Tuple[Optional[str], float]:
    try:
        closes = [float(k[4]) for k in khist]
        if len(closes) < max(fast, slow) + 1:
            return None, 0.0
        emas_fast = ema_from_list(closes, fast)
        emas_slow = ema_from_list(closes, slow)
        if emas_fast[-2] <= emas_slow[-2] and emas_fast[-1] > emas_slow[-1]:
            return "BUY", abs(emas_fast[-1]-emas_slow[-1])
        if emas_fast[-2] >= emas_slow[-2] and emas_fast[-1] < emas_slow[-1]:
            return "SELL", abs(emas_fast[-1]-emas_slow[-1])
        return None, 0.0
    except:
        return None, 0.0

def get_rsi(khist: List[List[Any]], period=RSI_PERIOD) -> Optional[float]:
    try:
        closes = [float(k[4]) for k in khist]
        if len(closes) < period + 1:
            return None
        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        gains_avg = sum(gains[-period:]) / period
        losses_avg = sum(losses[-period:]) / period
        if losses_avg == 0:
            return 100.0
        rs = gains_avg / losses_avg
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except:
        return None

def supertrend(khist: List[List[Any]], period=ST_PERIOD, multiplier=ST_MULTIPLIER) -> Optional[str]:
    try:
        if len(khist) < period + 2:
            return None
        highs = [float(k[2]) for k in khist]
        lows = [float(k[3]) for k in khist]
        closes = [float(k[4]) for k in khist]
        tr_list = []
        for i in range(1, len(khist)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            tr_list.append(tr)
        atr_val = sum(tr_list[-period:]) / period if len(tr_list) >= period else (sum(tr_list)/len(tr_list) if tr_list else 0.0)
        if atr_val == 0:
            return None
        hl2 = [(h + l) / 2 for h, l in zip(highs, lows)]
        upperband = [hl2[i] + multiplier * atr_val for i in range(len(hl2))]
        lowerband = [hl2[i] - multiplier * atr_val for i in range(len(hl2))]
        final_upper = [0]*len(hl2)
        final_lower = [0]*len(hl2)
        trend = [True]*len(hl2)
        for i in range(len(hl2)):
            if i == 0:
                final_upper[i] = upperband[i]
                final_lower[i] = lowerband[i]
                trend[i] = True if closes[i] > final_upper[i] else False
            else:
                final_upper[i] = min(upperband[i], final_upper[i-1]) if closes[i-1] <= final_upper[i-1] else upperband[i]
                final_lower[i] = max(lowerband[i], final_lower[i-1]) if closes[i-1] >= final_lower[i-1] else lowerband[i]
                if closes[i] > final_upper[i]:
                    trend[i] = True
                elif closes[i] < final_lower[i]:
                    trend[i] = False
                else:
                    trend[i] = trend[i-1]
        return "UP" if trend[-1] else "DOWN"
    except:
        return None

def bollinger_bands(khist: List[List[Any]], period=BB_PERIOD, std_mult=BB_STD) -> Dict[str,Any]:
    try:
        closes = [float(k[4]) for k in khist]
        if len(closes) < period:
            return {"upper":None,"mid":None,"lower":None,"width":None,"position":None}
        window = closes[-period:]
        mid = sum(window)/period
        variance = sum((c-mid)**2 for c in window)/period
        sd = math.sqrt(variance)
        upper = mid + std_mult * sd
        lower = mid - std_mult * sd
        width = (upper - lower) / mid if mid != 0 else None
        last = closes[-1]
        if last > upper: pos = "above_upper"
        elif last < lower: pos = "below_lower"
        else: pos = "inside"
        return {"upper":upper,"mid":mid,"lower":lower,"width":width,"position":pos}
    except:
        return {"upper":None,"mid":None,"lower":None,"width":None,"position":None}

# ---------------- AI heuristics ----------------
def ai_heuristics(khist: List[List[Any]], ob: Dict[str,Any],
                  st_val: Optional[str], rsi_val: Optional[float],
                  ema_side: Optional[str],
                  indicators_enabled: bool) -> Tuple[Optional[str], float]:
    """Heuristics; only use indicator boosts if indicators_enabled=True (i.e., timeframe == 15m)."""
    try:
        closes = [float(k[4]) for k in khist[-60:]] if len(khist)>=10 else [float(k[4]) for k in khist]
        highs = [float(k[2]) for k in khist[-60:]] if len(khist)>=10 else [float(k[2]) for k in khist]
        lows  = [float(k[3]) for k in khist[-60:]] if len(khist)>=10 else [float(k[3]) for k in khist]
        last = closes[-1] if closes else None
        if last is None:
            return None, 0.0

        recent_high = max(highs[-20:]) if highs else last
        recent_low = min(lows[-20:]) if lows else last
        bids = sum([float(x[1]) for x in ob.get("bids",[])[:15]]) if ob.get("bids") else 0
        asks = sum([float(x[1]) for x in ob.get("asks",[])[:15]]) if ob.get("asks") else 0
        total = bids+asks if (bids+asks)!=0 else 1.0
        imbalance = (bids-asks)/total

        # Base confidence from price action and order book
        conf_base = 0.0
        side = None
        if last > recent_high * 0.995:
            side, conf_base = "BUY", 0.6
        elif last < recent_low * 1.005:
            side, conf_base = "SELL", 0.6
        else:
            if imbalance > 0.1: side, conf_base = "BUY", 0.52
            elif imbalance < -0.1: side, conf_base = "SELL", 0.52

        if not side: return None, 0.0

        # Adjust confidence only if indicators_enabled (== True for 15m)
        indicator_conf_boost = 0.0
        if indicators_enabled:
            if side == "BUY":
                if st_val == "UP": indicator_conf_boost += 0.1
                if rsi_val and rsi_val < 50: indicator_conf_boost += 0.05
                if ema_side == "BUY": indicator_conf_boost += 0.1
            elif side == "SELL":
                if st_val == "DOWN": indicator_conf_boost += 0.1
                if rsi_val and rsi_val > 50: indicator_conf_boost += 0.05
                if ema_side == "SELL": indicator_conf_boost += 0.1

        conf = min(0.99, conf_base + abs(imbalance)*0.25 + indicator_conf_boost)
        return side, conf
    except:
        return None, 0.0

# ---------------- Combined strength ----------------
def combined_strength(ut_conf: float, lrc_conf: float) -> float:
    try:
        return W_UT * (ut_conf or 0) + W_LRC * (lrc_conf or 0)
    except:
        return 0.0

# ---------------- Merge decision ----------------
def merge_decision(sym: str, timeframe_minutes: int, khist: List[List[Any]]) -> Optional[Dict[str,Any]]:
    if not khist or len(khist) < timeframe_minutes+5: return None
    window = khist[-1:]
    ob = fetch_orderbook(sym)

    indicators_enabled = (timeframe_minutes == 15)  # <--- only 15m can use indicators

    ut_side = lrc_side = None
    ut_conf = lrc_conf = 0.0
    st_val = None; rsi_val = None; ema_side = None; ema_strength = 0.0; bb = {"upper":None,"mid":None,"lower":None,"width":None,"position":None}

    if indicators_enabled:
        ut_side, ut_conf = utbot_flags(khist)
        lrc_side, lrc_conf = linreg_flags(khist)
        st_val = supertrend(khist, period=ST_PERIOD, multiplier=ST_MULTIPLIER)
        rsi_val = get_rsi(khist, RSI_PERIOD)
        ema_side, ema_strength = ema_crossover(khist, EMA_FAST, EMA_SLOW)
        bb = bollinger_bands(khist, BB_PERIOD, BB_STD)

    ai_side, ai_conf = ai_heuristics(khist, ob, st_val, rsi_val, ema_side, indicators_enabled)

    # Combined strength from UT & LRC (only meaningful when indicators_enabled)
    strength_score = combined_strength(ut_conf or 0, lrc_conf or 0) if indicators_enabled else 0.0

    votes = []
    if indicators_enabled:
        if ut_side and ut_side!="NEUTRAL": votes.append(ut_side)
        if lrc_side: votes.append(lrc_side)
    ind_side = None
    if votes:
        if votes.count("BUY") > votes.count("SELL"): ind_side="BUY"
        elif votes.count("SELL") > votes.count("BUY"): ind_side="SELL"

    if not ai_side: return None

    # Label / merge rule:
    if indicators_enabled and ind_side is not None and ind_side == ai_side:
        final_side = ai_side; label="AI+INDICATORS"; conf=min(0.99, ai_conf+0.2 + 0.1*strength_score)
    elif indicators_enabled and ind_side is not None and ind_side != ai_side:
        return None  # disagreement on 15m -> reject
    else:
        # 5m path OR 15m with no indicator votes -> AI only
        final_side = ai_side; label="AI ONLY"; conf=ai_conf

    # Enforce minimum confidence
    if conf < MIN_CONF_POST: return None

    # Strict indicator filters apply ONLY on 15m
    if indicators_enabled:
        if st_val is not None:
            if final_side == "BUY" and st_val != "UP": return None
            if final_side == "SELL" and st_val != "DOWN": return None
        if ema_side is not None and ema_side != final_side:
            return None
        if rsi_val is not None:
            if final_side == "BUY" and rsi_val > 70: return None
            if final_side == "SELL" and rsi_val < 30: return None

        # ---------- Adaptive Bollinger Filter ----------
        # If price is above_upper or below_lower, decide based on trend vs sideways
        bb_pos = (bb or {}).get("position")
        # Determine trend: prefer Supertrend (UP/DOWN) then EMA (BUY/SELL)
        trend = None
        if st_val in ("UP","DOWN"):
            trend = st_val
        elif ema_side in ("BUY","SELL"):
            trend = "UP" if ema_side=="BUY" else "DOWN"

        # If BB above upper and signal is BUY -> block in sideways; allow only if trend==UP
        if bb_pos == "above_upper" and final_side == "BUY":
            if trend is None:
                return None
            else:
                if trend != "UP":
                    return None
        # If BB below lower and signal is SELL -> block in sideways; allow only if trend==DOWN
        if bb_pos == "below_lower" and final_side == "SELL":
            if trend is None:
                return None
            else:
                if trend != "DOWN":
                    return None
        # If bb inside -> ok. Other combinations (e.g., BUY when below_lower) are permitted by other filters.

    # Volatility & ranges
    highs=[float(k[2]) for k in khist[-60:]] if len(khist)>=60 else [float(k[2]) for k in khist]
    lows=[float(k[3]) for k in khist[-60:]] if len(khist)>=60 else [float(k[3]) for k in khist]
    atrv = sum([(h-l) for h,l in zip(highs,lows)]) / max(1,len(highs))
    entry = float(window[-1][4])
    avg_range = atrv if atrv>0 else max(1e-6, max(highs)-min(lows))
    vf = 1.0
    if avg_range < 0.5: vf=0.8
    elif avg_range > 2.0: vf=1.6

    atr_val = atr(khist, UTBOT_ATR_PERIOD)
    high_volatility = False
    if atr_val and entry > 0:
        vol_ratio = atr_val / entry
        high_volatility = vol_ratio >= 0.002
    if high_volatility:
        dyn_tp_mult = 2.5
        dyn_sl_mult = 1.8
    else:
        dyn_tp_mult = 4.0
        dyn_sl_mult = 3.0

    # ---------- Enforce R:R = 1:2 ----------
    # Compute a base SL distance (scaled by timeframe & volatility) then set TP distance = 2 * SL distance
    if timeframe_minutes == 15:
        base_sl_mult = SL_MULT_15M * (dyn_sl_mult / 1.8)
    elif timeframe_minutes == 5:
        base_sl_mult = SL_MULT_5M * (dyn_sl_mult / 1.8)
    else:
        base_sl_mult = SL_MULT_15M * (dyn_sl_mult / 1.8)

    sl_distance = base_sl_mult * avg_range * vf
    # safety fallback if computed distance is zero or tiny
    if sl_distance <= 0 or math.isnan(sl_distance):
        sl_distance = max(0.0001 * entry, 0.5 * avg_range)

    tp_distance = 2.0 * sl_distance

    if final_side=="BUY":
        tp = round(entry + tp_distance, 8)
        sl = round(entry - sl_distance, 8)
    else:
        tp = round(entry - tp_distance, 8)
        sl = round(entry + sl_distance, 8)

    # 24h quote volume check (best-effort, skip if not provided)
    tkr = fetch_ticker(sym)
    qvol = 0.0
    if isinstance(tkr, dict):
        info = tkr.get("info", {})
        for k in ("quoteVolume", "turnover24h", "volumeUsd24h"):
            v = info.get(k)
            if v is not None:
                try:
                    qvol = float(v); break
                except:
                    pass

    vol_level = volume_level(khist)

    return {"timeframe":f"{timeframe_minutes}m","label":label,"side":final_side,"confidence":conf,"entry":round(entry,8),
            "tp":tp,"sl":sl,"volume":vol_level,"ind_ut":ut_side,"ind_lrc":lrc_side,
            "rsi": rsi_val, "supertrend": st_val, "ema": ema_side, "high_volatility": high_volatility,
            "bollinger": bb, "strength_score": strength_score, "qvol": qvol}

# ---------------- Active trades ----------------
active_trades: Dict[str,Dict[str,Any]] = {}
active_lock = threading.Lock()

def record_trade(history_entry:Dict[str,Any], status:str="opened"):
    rec = {"ts":datetime.now(timezone.utc).isoformat(),"symbol":history_entry.get("symbol"),"timeframe":history_entry.get("timeframe"),
           "label":history_entry.get("label"),"side":history_entry.get("side"),"entry":history_entry.get("entry"),
           "tp":history_entry.get("tp"),"sl":history_entry.get("sl"),"confidence":history_entry.get("confidence"),
           "volume":history_entry.get("volume"),"ind_ut":history_entry.get("ind_ut"),"ind_lrc":history_entry.get("ind_lrc"),
           "status":status}
    db_push("trades_history", rec)

def update_accuracy(label:str, win:bool):
    stats = db_get("accuracy_stats", {"AI+INDICATORS":{"wins":0,"losses":0,"trades":0},"AI ONLY":{"wins":0,"losses":0,"trades":0}})
    key = label if label in stats else ("AI ONLY" if label=="AI ONLY" else "AI+INDICATORS")
    if win: stats[key]["wins"] += 1
    else: stats[key]["losses"] += 1
    stats[key]["trades"] += 1
    db_set("accuracy_stats", stats)

# ---------------- Model training (auto retrain) ----------------
USE_SKLEARN = True
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
except Exception:
    USE_SKLEARN = False
    print("scikit-learn not available — retrain will default to rule-based fallback.")

def features_from_record(rec):
    ind_ut = 1 if rec.get("ind_ut")=="BUY" else (-1 if rec.get("ind_ut")=="SELL" else 0)
    ind_lrc = 1 if rec.get("ind_lrc")=="BUY" else (-1 if rec.get("ind_lrc")=="SELL" else 0)
    conf = float(rec.get("confidence") or 0)
    vol = 1 if (rec.get("volume")=="High") else 0
    entry = float(rec.get("entry") or 0); tp=float(rec.get("tp") or 0); sl=float(rec.get("sl") or 0)
    atr_proxy = abs(tp-entry)+abs(entry-sl)
    side = 1 if rec.get("side")=="BUY" else 0
    bb = rec.get("bollinger") or {}
    bb_width = bb.get("width") or 0
    strength = rec.get("strength_score") or 0
    return [ind_ut, ind_lrc, conf, vol, atr_proxy, side, bb_width, strength]

def _collect_training_data_from_history():
    hist = db_get("trades_history", [])
    closed = [h for h in hist if str(h.get("status","")).startswith("closed")]
    filteredX=[]; filteredy=[]
    for r in closed:
        st = r.get("status")
        # NOTE: per request: include closed_be as loss (0) for training (do not skip)
        try:
            feat = features_from_record(r)
            filteredX.append(feat[:-1])
            filteredy.append(1 if st=="closed_win" else 0)
        except:
            pass
    return filteredX, filteredy

def _collect_seed_training_data_from_history():
    hist = db_get("trades_history", [])
    closed = [h for h in hist if str(h.get("status","")).startswith("closed")]
    first50 = closed[:50]
    X=[]; y=[]
    for r in first50:
        if r.get("status") != "closed_win": continue
        try:
            feat = features_from_record(r)
            X.append(feat[:-1]); y.append(1)
        except:
            continue
    if len(y) < 10:
        for r in closed:
            if r.get("status") == "closed_win":
                try:
                    feat = features_from_record(r)
                    X.append(feat[:-1]); y.append(1)
                except:
                    pass
    return X, y

def train_and_save_model(X, y):
    if not USE_SKLEARN or len(X) < 50:
        return None
    try:
        X = np.array(X); y = np.array(y)
        clf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)
        clf.fit(X,y)
        with open(MODEL_FILE,"wb") as f:
            pickle.dump(clf,f)
        print(f"Model trained on {len(X)} samples & saved.")
        return clf
    except Exception:
        safe_print_exc("train_and_save_model")
        return None

def train_model_if_needed():
    X,y = _collect_training_data_from_history()
    if len(y) >= 100:
        return train_and_save_model(X,y)
    return None

def load_model():
    if not USE_SKLEARN:
        return None
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE,"rb") as f:
                return pickle.load(f)
    except Exception:
        safe_print_exc("load_model")
    return None

MODEL = load_model()

# ---------------- Monitoring thread ----------------
def _bump_retrain_counter_and_maybe_retrain():
    try:
        cnt = int(db_get("retrain_counter", 0)) + 1
        db_set("retrain_counter", cnt)
        if cnt >= N_TRADES_FOR_RETRAIN:
            print("Retrain threshold reached. Attempting retrain...")
            new_model = train_and_save_model(*_collect_training_data_from_history())
            if new_model:
                global MODEL
                MODEL = new_model
                print("Retrain successful.")
            db_set("retrain_counter", 0)
    except Exception:
        safe_print_exc("retrain_counter")

def compute_estimated_pnl_usd(entry: float, target_price: float, margin_usd: float, leverage: float, side: str) -> float:
    """
    Integer-contract PnL (estimate):
    contracts = floor((margin * leverage) / entry)
    PnL = (target - entry) * contracts   (BUY)
    PnL = (entry - target) * contracts   (SELL)
    """
    try:
        pos_notional = margin_usd * leverage
        contracts = math.floor(pos_notional / entry) if entry > 0 else 0
        if contracts <= 0:
            return 0.0
        if side == "BUY":
            pnl = (target_price - entry) * contracts
        else:
            pnl = (entry - target_price) * contracts
        return float(pnl)
    except:
        return 0.0

def monitor_loop():
    while True:
        try:
            with active_lock:
                syms=list(active_trades.keys())
            for s in syms:
                try:
                    t = active_trades.get(s)
                    if not t: continue
                    price = fetch_last_price(s)
                    if price is None:
                        tk = fetch_ticker(s)
                        price = float(tk.get("last") or t.get("entry"))

                    if price is None or price <= 0: continue

                    entry = float(t["entry"]); tp = float(t["tp"]); sl = float(t["sl"]); side=t["side"]

                    hit_tp = (side=="BUY" and price>=tp) or (side=="SELL" and price<=tp)
                    hit_be = False
                    # BE confirmation: require partial_done + price within ±0.05% of entry + small time buffer since partial
                    if t.get("partial_done"):
                        partial_time = t.get("partial_time", 0)
                        # require at least 1 second since partial was recorded to avoid same-tick issues
                        if time.time() - partial_time > 1.0:
                            threshold = max(0.0005 * entry, 0.0000001 * entry)
                            if side == "BUY":
                                if price <= entry and abs(price-entry) <= threshold:
                                    hit_be = True
                            else:
                                if price >= entry and abs(price-entry) <= threshold:
                                    hit_be = True

                    hit_sl = (side=="BUY" and price<=sl) or (side=="SELL" and price>=sl)

                    # leverage from ccxt markets (max set earlier)
                    lev = fetch_and_set_max_leverage(s) or DEFAULT_LEVERAGE
                    pnl_tp = compute_estimated_pnl_usd(entry, tp, MARGIN_USD, lev, side)
                    pnl_sl = compute_estimated_pnl_usd(entry, sl, MARGIN_USD, lev, side)

                    if hit_tp:
                        send_telegram(f"Signal ({t['timeframe']}) - {t['label']}\nPair: {s}\nAction: {side}\nTarget Hit: TP\nPrice: {format_price(tp)} ✅\nProfit: (≈ ${pnl_tp:.2f})")
                        update_accuracy(t["label"], win=True)
                        record_trade({**t,"symbol":s}, status="closed_win")
                        with active_lock: active_trades.pop(s,None)
                        # set symbol-wide cooldown (both 5m and 15m) for COOLDOWN_MINUTES_AFTER_CLOSE
                        try:
                            cooldowns = db_get("cooldowns", {})
                            cooldown_until = int(time.time()*1000) + COOLDOWN_MINUTES_AFTER_CLOSE*60*1000
                            cooldowns[f"{s}|5m"] = cooldown_until
                            cooldowns[f"{s}|15m"] = cooldown_until
                            db_set("cooldowns", cooldowns)
                        except:
                            pass
                        _bump_retrain_counter_and_maybe_retrain()
                        continue

                    if hit_be:
                        send_telegram(f"Signal ({t['timeframe']}) - {t['label']}\nPair: {s}\nAction: {side}\nTarget Hit: BreakEven\nPrice: {format_price(entry)} ⚖️")
                        # Per request: count BreakEven as loss for retraining/accuracy stats
                        update_accuracy(t["label"], win=False)
                        record_trade({**t,"symbol":s}, status="closed_be")
                        with active_lock: active_trades.pop(s,None)
                        try:
                            cooldowns = db_get("cooldowns", {})
                            cooldown_until = int(time.time()*1000) + COOLDOWN_MINUTES_AFTER_CLOSE*60*1000
                            cooldowns[f"{s}|5m"] = cooldown_until
                            cooldowns[f"{s}|15m"] = cooldown_until
                            db_set("cooldowns", cooldowns)
                        except:
                            pass
                        _bump_retrain_counter_and_maybe_retrain()
                        continue

                    if hit_sl:
                        send_telegram(f"Signal ({t['timeframe']}) - {t['label']}\nPair: {s}\nAction: {side}\nTarget Hit: SL\nPrice: {format_price(sl)} ❌\nLoss: (≈ ${pnl_sl:.2f})")
                        update_accuracy(t["label"], win=False)
                        record_trade({**t,"symbol":s}, status="closed_loss")
                        with active_lock: active_trades.pop(s,None)
                        try:
                            cooldowns = db_get("cooldowns", {})
                            cooldown_until = int(time.time()*1000) + COOLDOWN_MINUTES_AFTER_CLOSE*60*1000
                            cooldowns[f"{s}|5m"] = cooldown_until
                            cooldowns[f"{s}|15m"] = cooldown_until
                            db_set("cooldowns", cooldowns)
                        except:
                            pass
                        _bump_retrain_counter_and_maybe_retrain()
                        continue

                    # Partial TP handling
                    if side=="BUY":
                        halfway = entry + (tp-entry)*PARTIAL_TP_RATIO
                        if price >= halfway and not t.get("partial_done"):
                            pnl_partial = compute_estimated_pnl_usd(entry, halfway, MARGIN_USD, fetch_and_set_max_leverage(s), side)
                            send_telegram(f"Signal ({t['timeframe']}) - {t['label']}\nPair: {s}\nAction: {side}\nPartial TP Hit: 50% booked, SL moved to BE\nPrice: {format_price(halfway)}\nEstimated P/L at Partial: (≈ ${pnl_partial:.2f})")
                            with active_lock:
                                active_trades[s]["partial_done"] = True
                                active_trades[s]["partial_time"] = time.time()
                                active_trades[s]["sl"] = entry
                    else:
                        halfway = entry - (entry - tp)*PARTIAL_TP_RATIO
                        if price <= halfway and not t.get("partial_done"):
                            pnl_partial = compute_estimated_pnl_usd(entry, halfway, MARGIN_USD, fetch_and_set_max_leverage(s), side)
                            send_telegram(f"Signal ({t['timeframe']}) - {t['label']}\nPair: {s}\nAction: {side}\nPartial TP Hit: 50% booked, SL moved to BE\nPrice: {format_price(halfway)}\nEstimated P/L at Partial: (≈ ${pnl_partial:.2f})")
                            with active_lock:
                                active_trades[s]["partial_done"] = True
                                active_trades[s]["partial_time"] = time.time()
                                active_trades[s]["sl"] = entry

                except Exception:
                    safe_print_exc("monitor_inner")

            time.sleep(CHECK_INTERVAL_MONITOR)
        except Exception:
            safe_print_exc("monitor_loop")
            time.sleep(5)

# ---------------- Analysis / Signal posting ----------------
def _is_on_cooldown(sym: str, timeframe_label: str) -> bool:
    try:
        cooldowns = db_get("cooldowns", {})
        key = f"{sym}|{timeframe_label}"
        until_ms = cooldowns.get(key)
        if not until_ms:
            return False
        return int(time.time()*1000) < int(until_ms)
    except:
        return False

def analyze_and_post(sym: str, minute:int, model):
    try:
        valid_timeframes = []
        if minute % 5 == 0:
            valid_timeframes.append("5m")
        if minute % 15 == 0:
            valid_timeframes.append("15m")

        if not valid_timeframes:
            print(f"[{datetime.utcnow().isoformat()}] HOLD - Waiting for a valid candle close time.")
            return

        for tf in valid_timeframes:
            if sym in active_trades:
                print(f"[{datetime.utcnow().isoformat()}] SKIP - active trade already exists for {sym}")
                continue

            # cooldown check per symbol/timeframe
            if _is_on_cooldown(sym, tf):
                print(f"[{datetime.utcnow().isoformat()}] SKIP - {sym} {tf} is on cooldown")
                continue

            # 24h volume filter (best-effort)
            pub = fetch_ticker(sym)
            qvol = 0.0
            if pub and isinstance(pub, dict):
                info = pub.get("info", {})
                for k in ("quoteVolume", "turnover24h", "volumeUsd24h"):
                    v = info.get(k)
                    if v is not None:
                        try:
                            qvol = float(v); break
                        except:
                            pass
            if qvol and qvol < MIN_24H_VOL_USDT:
                print(f"[{datetime.utcnow().isoformat()}] SKIP - Low 24h vol for {sym} at {tf}: {qvol}")
                continue

            khist = fetch_klines(sym, interval=tf, limit=300)
            sig = merge_decision(sym, int(tf.replace('m','')), khist)
            if not sig:
                print(f"[{datetime.utcnow().isoformat()}] HOLD - No trade signal for {sym} on {tf} timeframe")
                continue

            # Multi-timeframe opposition rejection (unchanged)
            other_tf = "15m" if tf == "5m" else "5m"
            try:
                khist_other = fetch_klines(sym, interval=other_tf, limit=300)
                other_sig = merge_decision(sym, int(other_tf.replace('m','')), khist_other)
                if other_sig and other_sig.get("side") and sig.get("side") and other_sig.get("side") != sig.get("side"):
                    print(f"[{datetime.utcnow().isoformat()}] REJECT - Opposite signals 5m vs 15m for {sym} ({tf} vs {other_tf})")
                    continue
            except Exception:
                pass

            # Model probability threshold (unchanged)
            if model and USE_SKLEARN:
                feat = [1 if sig.get("ind_ut")=="BUY" else (-1 if sig.get("ind_ut")=="SELL" else 0),
                        1 if sig.get("ind_lrc")=="BUY" else (-1 if sig.get("ind_lrc")=="SELL" else 0),
                        sig.get("confidence",0),
                        1 if sig.get("volume")=="High" else 0,
                        abs(sig.get("tp") - sig.get("entry")) + abs(sig.get("entry") - sig.get("sl")),
                        (sig.get("bollinger") or {}).get("width") or 0,
                        sig.get("strength_score") or 0]
                try:
                    prob = model.predict_proba([feat])[0][1]
                    if prob < MIN_CONF_POST:
                        print(f"[{sym}] model low prob on {tf}: {prob:.2f} -> skipping signal")
                        continue
                except Exception:
                    pass

            # AI ONLY allowed on 5m & 15m; AI+INDICATORS only possible on 15m (already enforced upstream)
            if sig.get("label") == "AI ONLY" and sig.get("confidence",0) < MIN_CONF_POST:
                print(f"[{datetime.utcnow().isoformat()}] SKIP - AI ONLY low confidence for {sym}")
                continue

            # Make sure leverage is set to max for this symbol (once)
            lev = fetch_and_set_max_leverage(sym) or DEFAULT_LEVERAGE

            # final sanity re-check cooldown again under lock and post
            with active_lock:
                if sym in active_trades: continue
                conf_label = "High" if sig['confidence']>=HIGH_CONF else "Medium"
                pnl_tp = compute_estimated_pnl_usd(sig['entry'], sig['tp'], MARGIN_USD, lev, sig['side'])
                pnl_sl = compute_estimated_pnl_usd(sig['entry'], sig['sl'], MARGIN_USD, lev, sig['side'])
                msg = (
                    f"Signal ({sig['timeframe']}) - {sig['label']}\n"
                    f"Pair: {sym}\nAction: {sig['side']}\nCurrent price: {format_price(sig['entry'])}\n"
                    f"FullTP: {format_price(sig['tp'])} (≈ ${pnl_tp:.2f})\nSL: {format_price(sig['sl'])} (≈ ${pnl_sl:.2f})\n"
                    f"Confidence: ({conf_label})\nVolume: ({sig['volume']})"
                )
                send_telegram(msg)
                active_trades[sym] = {"symbol":sym,"timeframe":sig["timeframe"],"label":sig["label"],"side":sig["side"],
                                      "entry":sig["entry"],"tp":sig["tp"],"sl":sig["sl"],"confidence":sig["confidence"],
                                      "volume":sig["volume"],"ind_ut":sig.get("ind_ut"),"ind_lrc":sig.get("ind_lrc"),
                                      "open_time":datetime.now(timezone.utc).isoformat(), "partial_done": False}
            record_trade({**active_trades[sym],"symbol":sym}, status="opened")
            stats = db_get("accuracy_stats")
            if sig["label"] in stats:
                stats[sig["label"]]["trades"] += 1
                db_set("accuracy_stats", stats)

    except Exception:
        safe_print_exc("analyze_and_post")

# ---------------- Startup seed training ----------------
def seed_train_on_start():
    try:
        X,y = _collect_seed_training_data_from_history()
        if len(y) >= 50:
            print(f"Seed-training from existing history (first50 wins): {len(y)} samples")
            clf = train_and_save_model(X,y)
            return clf
        else:
            X2,y2 = _collect_training_data_from_history()
            if len(y2) >= 100:
                print(f"Seed-training from existing history (fallback full): {len(y2)} samples")
                clf = train_and_save_model(X2,y2)
                return clf
            else:
                print(f"Seed-training skipped (found {len(y)} preferred wins and {len(y2)} fallback samples).")
                return None
    except Exception:
        safe_print_exc("seed_train_on_start")
        return None

# ---------------- Main ----------------
def main():
    global MODEL
    send_telegram("Scalping Bot (Futures) Started Successfully.")
    print("Bot started. Loading model...")
    if MODEL is None:
        print("No preloaded model. Trying seed-training...")
        maybe = seed_train_on_start()
        if maybe:
            MODEL = maybe
        else:
            print("Seed-training not performed.")

    # preload markets to speed symbol selection & leverage setting
    try:
        exchange.load_markets()
    except Exception:
        safe_print_exc("load_markets_on_start")

    mon = threading.Thread(target=monitor_loop, daemon=True)
    mon.start()

    dynamic_list = []
    try:
        all_symbols = fetch_top_n_symbols(500)
        chosen=[]
        for s in all_symbols:
            if s in SYMBOLS_MANDATORY: continue
            chosen.append(s)
            if len(chosen) >= DYNAMIC_TOP_N:
                break
        dynamic_list = chosen
    except Exception:
        safe_print_exc("dynamic_load")

    SYMBOLS = SYMBOLS_MANDATORY + dynamic_list
    print("Monitoring symbols:", SYMBOLS)

    # Pre-set leverage to max for the monitored set (best-effort, non-blocking)
    for sym in SYMBOLS:
        try:
            fetch_and_set_max_leverage(sym)
        except Exception:
            pass

    while True:
        try:
            now = datetime.utcnow()
            minute = now.minute
            for sym in SYMBOLS:
                analyze_and_post(sym, minute, MODEL)
                time.sleep(0.2)  # light pacing to avoid rate limits
        except Exception:
            safe_print_exc("main_loop")
        # tight loop per spec (5s)
        to_sleep = CHECK_INTERVAL_MAIN - (time.time() % CHECK_INTERVAL_MAIN)
        if to_sleep < 0.05:
            to_sleep = 0.05
        time.sleep(to_sleep)

if __name__ == "__main__":
    try:
        MODEL = load_model() if 'load_model' in globals() else None
    except Exception:
        MODEL = None
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception:
        safe_print_exc("startup")
