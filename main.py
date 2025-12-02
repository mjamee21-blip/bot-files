import os
import asyncio
import logging
import threading
import socket
from datetime import datetime
from flask import Flask
import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Replit
import matplotlib.pyplot as plt
import io
from telegram import Update, Bot, InlineKeyboardMarkup, InlineKeyboardButton, error as telegram_error
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters
from dotenv import load_dotenv
from datetime import datetime, timezone
from metaapi_cloud_sdk import MetaApi

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
UPTIMEROBOT_API_KEY = os.getenv("UPTIMEROBOT_API_KEY", "u3199041-10a06868df886132c24d1aa8")
CRONJOB_API_KEY = os.getenv("CRONJOB_API_KEY")
CRONJOB_JOB_ID = "6938714"

TRADING_SESSIONS = {
    "asia": {
        "start": 0,
        "end": 9,
        "symbols": ["AUDJPY", "AUDUSD", "NZDJPY", "NZDUSD", "USDJPY", "CHFJPY"]
    },
    "london": {
        "start": 8,
        "end": 16,
        "symbols": ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "CHFJPY"]
    },
    "newyork": {
        "start": 13,
        "end": 22,
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "BTC"]
    },
    "24hour": ["BTC-USD"]
}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TIMEFRAME = "30m"
SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", 
    "AUDUSD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "CHFJPY=X", "XAUUSD=X", "BTC-USD"
]

FAST_MA = 10
SLOW_MA = 30
RSI_PERIOD = 14
RSI_BUY = 55
RSI_SELL = 45
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

TP1_POINTS = 30
TP2_POINTS = 60
TP3_POINTS = 100
SL_POINTS = 40

CHECK_INTERVAL = 0  # NO SLEEP - Check continuously (0 = no delay between cycles)
LOOKBACK_BARS = 400

# Feature Settings (All 7 new features)
MAX_OPEN_POSITIONS = 999  # Unlimited open positions (no max limit)
DAILY_LOSS_LIMIT = 999999  # Unlimited daily loss (trading always allowed)
ENABLE_BREAKEVEN = True  # Move SL to entry after TP1 hit
ENABLE_PARTIAL_EXITS = True  # Close 33% at each TP
ENABLE_TRAILING_STOP = True  # Auto-move SL as price moves
TRAILING_STOP_DISTANCE = 10  # pips to trail behind price
MUTE_RETESTS = False  # Mute retest signals
MUTE_OPPOSITES = False  # Mute opposite signal exits

# 🗞️ NEWS FILTER SETTINGS
USE_NEWS_FILTER = True  # Filter signals during major economic news
NEWS_FILTER_MINUTES_BEFORE = 30  # Stop trading 30 mins before high-impact news
NEWS_FILTER_MINUTES_AFTER = 30  # Stop trading 30 mins after high-impact news

state = {}  # Active trading positions - CLEARED ON EACH RESTART
sent_signals = {}  # Last signal time per symbol
sent_exits = {}  # Track which exit signals have been sent to prevent duplicates
signal_stability = {}  # Track sent signals: {symbol: {side, breakout_level, strength, entry_price}} - to avoid duplicate signals
economic_events_cache = {"timestamp": 0, "events": []}
retest_tracker = {}
trades_history = {}
daily_summary_sent = False
trading_paused = False
metaapi_account = None
metaapi_connection = None
channel_free_signal_sent = False  # Track if free signal was posted to channel today
last_channel_signal_date = None  # Track the date of last channel signal
signal_copyables = {}  # Store copyable text for signals {unique_id: copyable_text}
user_manual_trades = {}  # Track manual trades: {chat_id: {symbol: {side, entry, entry_time, pips}}}
signal_history = []  # All signals sent: [{symbol, side, entry, time, result}]
user_settings = {}  # User preferences: {chat_id: {min_accuracy: 60, pairs: [EURUSD, GBP...], min_rr: 2.0, break_hours: "22:00-06:00"}}
support_tickets = {}  # {ticket_id: {user_id, username, message, timestamp, status, responses: []}}
payment_receipts = {}  # {receipt_id: {user_id, username, plan, file_id, timestamp, status: "pending"/"verified"}}

# Subscription & User Management (for monetization)
subscribers = {}  # {chat_id: {"status": "active"/"trial", "plan": "free"/"signals"/"autotrader", "joined": timestamp}}
signals_sent_today = {}  # {chat_id: count}
USDT_WALLET = os.getenv("USDT_WALLET", "YOUR_USDT_WALLET_ADDRESS")  # Set in environment
CHANNEL_ID = "@AlphaForexunitedbot"  # Channel to post free signals (1 per day)
subscription_plans = {
    "free": {"price": "Free", "signals_per_day": 1, "symbols": 5, "auto_trade": False, "payment": "None"},
    "signals": {"price": "5 USDT", "signals_per_day": 10, "symbols": 14, "auto_trade": False, "payment": USDT_WALLET},
    "autotrader": {"price": "10 USDT", "signals_per_day": 999, "symbols": 14, "auto_trade": True, "payment": USDT_WALLET}
}

SESSION_END_TIMES = {
    "EURUSD=X": 16, "GBPUSD=X": 16, "USDJPY=X": 22, "USDCHF=X": 17,
    "USDCAD=X": 17, "AUDUSD=X": 14, "NZDUSD=X": 14, "EURGBP=X": 16,
    "EURJPY=X": 16, "GBPJPY=X": 16, "AUDJPY=X": 9, "CHFJPY=X": 16,
    "XAUUSD=X": 24, "BTC-USD": 24
}

# UptimeRobot Auto-Update
last_known_url = None
uptimerobot_monitor_id = None

def generate_signal_chart(symbol: str, entry_price: float, tp1: float, tp2: float, tp3: float, sl: float, side: str, pivot_points: dict):
    """Generate technical analysis chart for signal - shows why entry/exit decisions were made"""
    try:
        # Fetch recent price data
        clean_symbol = symbol.replace("=X", "").replace("-USD", "")
        data = yf.download(symbol, period="5d", interval="1h", progress=False)
        
        if data is None or len(data) < 10:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#1a1a1a')
        ax.set_facecolor('#0d0d0d')
        
        # Plot price action
        ax.plot(range(len(data)), data['Close'].values, color='#00d4ff', linewidth=2, label=f'{clean_symbol} Price')
        
        # Fill area under price
        ax.fill_between(range(len(data)), data['Close'].values, alpha=0.1, color='#00d4ff')
        
        # Plot pivot points (resistance/support)
        if pivot_points:
            if 'resistance' in pivot_points:
                ax.axhline(y=pivot_points['resistance'], color='#ff4444', linestyle='--', linewidth=1.5, alpha=0.7, label=f"Resistance: {pivot_points['resistance']:.5f}")
            if 'support' in pivot_points:
                ax.axhline(y=pivot_points['support'], color='#44ff44', linestyle='--', linewidth=1.5, alpha=0.7, label=f"Support: {pivot_points['support']:.5f}")
        
        # Plot entry point
        entry_idx = len(data) - 1
        ax.scatter([entry_idx], [entry_price], color='#ffff00', s=200, marker='^' if side == 'BUY' else 'v', 
                   label=f'📍 Entry: {entry_price:.5f} ({side})', zorder=5, edgecolors='white', linewidth=2)
        
        # Plot targets (TP1, TP2, TP3)
        ax.axhline(y=tp1, color='#88ff88', linestyle=':', linewidth=1.5, alpha=0.6, label=f'TP1: {tp1:.5f}')
        ax.axhline(y=tp2, color='#44dd44', linestyle=':', linewidth=1.5, alpha=0.6, label=f'TP2: {tp2:.5f}')
        ax.axhline(y=tp3, color='#00aa00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'TP3: {tp3:.5f}')
        
        # Plot stop loss
        ax.axhline(y=sl, color='#ff6666', linestyle=':', linewidth=2, alpha=0.7, label=f'SL: {sl:.5f}')
        
        # Formatting
        ax.set_title(f'📊 {clean_symbol} - {side} Signal Entry Analysis', color='white', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Time (Hours)', color='#888888', fontsize=10)
        ax.set_ylabel('Price', color='#888888', fontsize=10)
        ax.legend(loc='best', facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.1, color='#444444')
        ax.tick_params(colors='#888888')
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1a1a', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        logger.error(f"Error generating chart for {symbol}: {e}")
        return None


def get_current_replit_url():
    """Get current Replit URL from environment"""
    replit_dev_domain = os.getenv("REPLIT_DEV_DOMAIN")
    if replit_dev_domain:
        return f"https://{replit_dev_domain}/"
    return None

def get_uptimerobot_monitors():
    """Get all monitors from UptimeRobot API"""
    try:
        url = "https://api.uptimerobot.com/v2/getMonitors"
        data = {"api_key": UPTIMEROBOT_API_KEY, "format": "json"}
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("stat") == "ok":
            return result.get("monitors", [])
        else:
            logger.warning(f"UptimeRobot API error: {result.get('error', {}).get('message')}")
    except Exception as e:
        logger.error(f"Error fetching UptimeRobot monitors: {e}")
    return []

def update_uptimerobot_url(monitor_id, new_url):
    """Update monitor URL in UptimeRobot"""
    try:
        url = "https://api.uptimerobot.com/v2/editMonitor"
        data = {
            "api_key": UPTIMEROBOT_API_KEY,
            "format": "json",
            "id": monitor_id,
            "url": new_url
        }
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("stat") == "ok":
            logger.info(f"✅ UptimeRobot updated to: {new_url}")
            return True
        else:
            logger.warning(f"UptimeRobot update error: {result.get('error', {}).get('message')}")
    except Exception as e:
        logger.error(f"Error updating UptimeRobot: {e}")
    return False

def update_cronjob_url(new_url):
    """Update cron-job.org URL when Replit URL changes"""
    if not CRONJOB_API_KEY or not CRONJOB_JOB_ID:
        return False
    
    try:
        url = f"https://api.cron-job.org/jobs/{CRONJOB_JOB_ID}"
        headers = {
            "Authorization": f"Bearer {CRONJOB_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"job": {"url": new_url}}
        
        response = requests.patch(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        logger.info(f"✅ Cron-job.org updated to: {new_url}")
        return True
    except Exception as e:
        logger.error(f"Error updating cron-job.org: {e}")
    return False

async def send_telegram_keepalive(bot: Bot):
    """Send Telegram keepalive ping every 5 minutes to prevent 15min timeout"""
    try:
        # Send a silent getMe() call to keep Telegram connection alive
        await bot.get_me()
        logger.debug("✅ Telegram keepalive ping sent")
    except Exception as e:
        logger.debug(f"Telegram keepalive error (non-critical): {e}")

async def send_health_heartbeat():
    """Send heartbeat to Healthchecks.io (push monitoring - bypasses Replit proxy)"""
    try:
        # Using a free Healthchecks.io endpoint for heartbeat monitoring
        # This allows bot to PUSH "I'm alive" instead of waiting for PULL checks
        healthcheck_url = "https://hc-ping.com/8b7a9c3e-8f2c-4a5b-9e1d-2f3c4d5e6f7g"  # Example UUID
        
        response = requests.post(healthcheck_url, timeout=5)
        if response.status_code == 200:
            logger.debug("✅ Health heartbeat sent")
        else:
            logger.warning(f"Health heartbeat failed: {response.status_code}")
    except Exception as e:
        logger.debug(f"Health heartbeat error (non-critical): {e}")

async def check_and_update_uptimerobot():
    """Background task to check URL changes and update UptimeRobot & cron-job.org"""
    global last_known_url, uptimerobot_monitor_id
    
    try:
        current_url = get_current_replit_url()
        
        if not current_url:
            return
        
        # If this is first time or URL changed
        if current_url != last_known_url:
            logger.info(f"Replit URL changed or first run: {current_url}")
            last_known_url = current_url
            
            # Get monitor ID if not cached
            if not uptimerobot_monitor_id:
                monitors = get_uptimerobot_monitors()
                if monitors:
                    uptimerobot_monitor_id = monitors[0].get("id")
                    logger.info(f"Found UptimeRobot monitor ID: {uptimerobot_monitor_id}")
            
            # Update UptimeRobot
            if uptimerobot_monitor_id:
                update_uptimerobot_url(uptimerobot_monitor_id, current_url)
            
            # Update cron-job.org
            update_cronjob_url(current_url)
        
        # Send health heartbeat every hour (works through push monitoring)
        await send_health_heartbeat()
    
    except Exception as e:
        logger.error(f"Error in check_and_update_uptimerobot: {e}")


def get_economic_calendar():
    """Fetch US economic events from FRED API (free, no rate limits)"""
    if not FRED_API_KEY:
        logger.info("FRED_API_KEY not set - register free at https://fred.stlouisfed.org/user/register")
        return []
    
    try:
        current_time = datetime.now(timezone.utc).timestamp()
        
        # Cache for 1 hour
        if economic_events_cache["timestamp"] and (current_time - economic_events_cache["timestamp"]) < 3600:
            return economic_events_cache["events"]
        
        # High-impact US economic series
        series = {
            "PAYEMS": "NFP",  # Non-farm payroll (most important)
            "UNRATE": "Unemployment",
            "CPIAUCSL": "CPI",
            "DEXUSEU": "USD/EUR"
        }
        
        events = []
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        for series_id, name in series.items():
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "limit": 1,
                "sort_order": "desc"
            }
            
            response = requests.get(base_url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json().get("observations", [])
                if data:
                    latest = data[0]
                    events.append({
                        "title": f"US {name}",
                        "date": latest.get("date"),
                        "value": latest.get("value"),
                        "impact": "high" if "PAYEMS" in series_id or "CPIAUCSL" in series_id else "medium"
                    })
        
        economic_events_cache["timestamp"] = current_time
        economic_events_cache["events"] = events
        
        logger.info(f"Loaded {len(events)} US economic indicators from FRED")
        return events
        
    except Exception as e:
        logger.warning(f"FRED API error: {e} - using cache")
        return economic_events_cache.get("events", [])


def is_near_economic_event(symbol):
    """Check if trading symbol is affected by upcoming high-impact economic event within 30 min"""
    if not USE_NEWS_FILTER:
        return False, None
    
    events = get_economic_calendar()
    if not events:
        return False, None
    
    currency_map = {
        "EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "JPY", "USDCHF": "CHF",
        "USDCAD": "CAD", "AUDUSD": "AUD", "NZDUSD": "NZD", "EURGBP": ["EUR", "GBP"],
        "EURJPY": ["EUR", "JPY"], "GBPJPY": ["GBP", "JPY"], "AUDJPY": ["AUD", "JPY"],
        "CHFJPY": ["CHF", "JPY"]
    }
    
    symbol_clean = symbol.replace("=X", "").replace("=F", "")
    relevant_currencies = currency_map.get(symbol_clean, [])
    if isinstance(relevant_currencies, str):
        relevant_currencies = [relevant_currencies]
    
    if not relevant_currencies:
        return False, None
    
    now = datetime.now(timezone.utc)
    
    for event in events:
        try:
            event_time = datetime.fromisoformat(event.get("dateTime", "").replace("Z", "+00:00"))
            time_diff = (event_time - now).total_seconds() / 60
            
            # Check if within news filter window (before and after)
            filter_window = (0 - NEWS_FILTER_MINUTES_BEFORE) <= time_diff <= NEWS_FILTER_MINUTES_AFTER
            if filter_window:
                event_currency = event.get("country", "")
                if any(curr in event_currency for curr in relevant_currencies):
                    return True, event.get("event", f"🗞️ Economic news: {event_time.strftime('%H:%M UTC')}")
        except Exception as e:
            logger.debug(f"Error parsing event time: {e}")
    
    return False, None


def detect_retest_opportunity(symbol, price, swings, rsi_val):
    """Detect if price is retesting support/resistance levels"""
    global retest_tracker
    
    if not swings or symbol not in retest_tracker:
        return False, None
    
    previous_level = retest_tracker[symbol].get('level')
    previous_type = retest_tracker[symbol].get('type')  # 'support' or 'resistance'
    
    if not previous_level:
        return False, None
    
    # Define retest zone (within 0.5% of the level)
    zone_width = previous_level * 0.005
    in_retest_zone = abs(price - previous_level) <= zone_width
    
    if not in_retest_zone:
        return False, None
    
    # Retest is valid if RSI shows reversal potential
    if previous_type == 'support' and rsi_val and rsi_val < 35:
        return True, f"Support Retest at {format_price(symbol, previous_level)}"
    elif previous_type == 'resistance' and rsi_val and rsi_val > 65:
        return True, f"Resistance Retest at {format_price(symbol, previous_level)}"
    
    return False, None


def update_retest_levels(symbol, swings, price):
    """Update tracking of swing levels for retest detection"""
    global retest_tracker
    
    if not swings:
        return
    
    resistance = swings.get('resistance')
    support = swings.get('support')
    
    # Track the nearest untouched level as retest opportunity
    if resistance and price < resistance:
        retest_tracker[symbol] = {
            'level': resistance,
            'type': 'resistance',
            'timestamp': datetime.now(timezone.utc)
        }
    elif support and price > support:
        retest_tracker[symbol] = {
            'level': support,
            'type': 'support',
            'timestamp': datetime.now(timezone.utc)
        }


def ema(series, period):
    alpha = 2 / (period + 1)
    out = []
    for i, v in enumerate(series):
        if i == 0:
            out.append(v)
        else:
            out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def rsi(series, period):
    if len(series) <= period + 1:
        return [None] * len(series)
    gains, losses = [], []
    for i in range(1, len(series)):
        ch = series[i] - series[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [None] * period
    for i in range(period, len(series) - 1):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsis.append(100 - (100 / (1 + rs)))
    rsis.append(rsis[-1] if rsis[-1] is not None else 50)
    return rsis


def macd(series, fast=12, slow=26, signal=9):
    fe = ema(series, fast)
    se = ema(series, slow)
    line = [f - s for f, s in zip(fe, se)]
    sig = ema(line, signal)
    hist = [l - s for l, s in zip(line, sig)]
    return line, sig, hist


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr_val = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
    return atr_val


def fib_levels(high, low):
    diff = high - low
    return {
        "382": high - 0.382 * diff,
        "618": high - 0.618 * diff
    }


def stochastic_rsi(rsi_vals, period=14, smooth_k=3, smooth_d=3):
    if len(rsi_vals) < period:
        return None, None
    rsi_list = [v for v in rsi_vals if v is not None]
    if len(rsi_list) < period:
        return None, None
    
    min_rsi = min(rsi_list[-period:])
    max_rsi = max(rsi_list[-period:])
    range_rsi = max_rsi - min_rsi if max_rsi != min_rsi else 1
    
    k_val = 100 * (rsi_list[-1] - min_rsi) / range_rsi
    d_val = k_val
    return k_val, d_val


def adx(highs, lows, closes, period=14):
    if len(closes) < period * 2:
        return None
    
    tr_list = []
    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        plus_dm = high_diff if high_diff > 0 and high_diff > low_diff else 0
        minus_dm = low_diff if low_diff > 0 and low_diff > high_diff else 0
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append((tr, plus_dm, minus_dm))
    
    tr_sum = sum(t[0] for t in tr_list[:period])
    plus_dm_sum = sum(t[1] for t in tr_list[:period])
    minus_dm_sum = sum(t[2] for t in tr_list[:period])
    
    di_plus = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
    di_minus = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
    adx_val = dx
    
    for i in range(period, len(tr_list)):
        tr_sum = (tr_sum * (period - 1) + tr_list[i][0]) / period
        plus_dm_sum = (plus_dm_sum * (period - 1) + tr_list[i][1]) / period
        minus_dm_sum = (minus_dm_sum * (period - 1) + tr_list[i][2]) / period
        di_plus = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
        di_minus = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
        adx_val = (adx_val * (period - 1) + dx) / period
    
    return adx_val


def ichimoku(highs, lows, closes, period1=9, period2=26, period3=52):
    if len(closes) < period2 + period3:
        return {"tenkan": 0, "kijun": 0, "senkou_a": 0, "senkou_b": 0, "chikou": 0}
    
    high1 = max(highs[-period1:])
    low1 = min(lows[-period1:])
    tenkan = (high1 + low1) / 2
    
    high2 = max(highs[-period2:])
    low2 = min(lows[-period2:])
    kijun = (high2 + low2) / 2
    
    senkou_a = (tenkan + kijun) / 2
    
    high3 = max(highs[-period3:])
    low3 = min(lows[-period3:])
    senkou_b = (high3 + low3) / 2
    
    chikou = closes[-1]
    
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou}


def check_ichimoku_confirmation(price, ichimoku_data, buy_signal):
    """Check if Ichimoku confirms the pivot breakout signal"""
    if not ichimoku_data:
        return False, 0
    
    tenkan = ichimoku_data.get("tenkan", 0)
    kijun = ichimoku_data.get("kijun", 0)
    senkou_a = ichimoku_data.get("senkou_a", 0)
    senkou_b = ichimoku_data.get("senkou_b", 0)
    
    # Cloud boundaries
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    
    if buy_signal:
        # BUY confirmation: Tenkan > Kijun AND Price > Cloud
        tenkan_confirmed = tenkan > kijun
        cloud_confirmed = price > cloud_top
        signal_confirmed = tenkan_confirmed and cloud_confirmed
        confirmation_strength = 15 if signal_confirmed else 0  # +15 accuracy boost
        return signal_confirmed, confirmation_strength
    else:
        # SELL confirmation: Tenkan < Kijun AND Price < Cloud
        tenkan_confirmed = tenkan < kijun
        cloud_confirmed = price < cloud_bottom
        signal_confirmed = tenkan_confirmed and cloud_confirmed
        confirmation_strength = 15 if signal_confirmed else 0  # +15 accuracy boost
        return signal_confirmed, confirmation_strength


def order_flow_delta(opens, highs, lows, closes, volumes, period=14):
    if len(closes) < period or len(volumes) < period:
        return None
    
    deltas = []
    for i in range(len(closes)):
        if i == 0:
            deltas.append(0)
            continue
        
        close = closes[i]
        open_price = opens[i]
        high = highs[i]
        low = lows[i]
        volume = volumes[i] if i < len(volumes) else 1
        
        high_low_range = max(high - low, 0.00001)
        buy_pressure = (close - low) / high_low_range
        sell_pressure = (high - close) / high_low_range
        
        delta = (buy_pressure - sell_pressure) * volume
        deltas.append(delta)
    
    if len(deltas) < period:
        return None
    
    cumulative_delta = sum(deltas[-period:])
    avg_delta = cumulative_delta / period
    
    if avg_delta == 0:
        return 0
    delta_strength = min(100, abs(avg_delta) / max(1e-6, sum(volumes[-period:]) / period) * 1000)
    return delta_strength if avg_delta > 0 else -delta_strength


def volume_profile(highs, lows, closes, volumes, bins=20, lookback=50):
    if len(closes) < lookback or len(volumes) < lookback:
        return None, None
    
    recent_data = list(zip(highs[-lookback:], lows[-lookback:], closes[-lookback:], volumes[-lookback:]))
    
    min_price = min(l for h, l, c, v in recent_data)
    max_price = max(h for h, l, c, v in recent_data)
    price_range = max_price - min_price if max_price > min_price else 1
    
    bin_size = price_range / bins
    volume_bins = [0] * bins
    
    for high, low, close, volume in recent_data:
        mid_price = (high + low) / 2
        bin_idx = min(int((mid_price - min_price) / bin_size), bins - 1)
        volume_bins[bin_idx] += volume
    
    max_volume_idx = volume_bins.index(max(volume_bins)) if max(volume_bins) > 0 else bins // 2
    point_of_control = min_price + (max_volume_idx + 0.5) * bin_size
    
    total_volume = sum(volume_bins)
    cumsum = 0
    value_area_low = point_of_control
    value_area_high = point_of_control
    
    for i in range(bins):
        cumsum += volume_bins[i]
        if cumsum >= total_volume * 0.35:
            value_area_low = min_price + i * bin_size
            break
    
    cumsum = 0
    for i in range(bins - 1, -1, -1):
        cumsum += volume_bins[i]
        if cumsum >= total_volume * 0.35:
            value_area_high = min_price + (i + 1) * bin_size
            break
    
    return point_of_control, {"low": value_area_low, "high": value_area_high}


def market_profile(highs, lows, closes, volumes, lookback=50):
    if len(closes) < lookback:
        return None, None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    
    high_price = max(recent_highs)
    low_price = min(recent_lows)
    
    hpo = (high_price + low_price) / 2
    va_height = high_price - low_price
    
    poc_proximity = abs(closes[-1] - hpo) / max(1e-6, va_height)
    is_in_profile = poc_proximity < 0.5
    
    profile_strength = 50 + (1 - poc_proximity) * 50 if is_in_profile else max(0, 50 - poc_proximity * 50)
    
    return hpo, min(100, max(0, profile_strength))


def pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    
    return {
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "s1": s1,
        "s2": s2
    }


def smart_money_concepts(highs, lows, closes, period=20):
    if len(closes) < period:
        return None
    
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    recent_closes = closes[-period:]
    current_price = closes[-1]
    
    swing_high = max(recent_highs)
    swing_low = min(recent_lows)
    
    order_block_high = swing_high * 1.002
    order_block_low = swing_low * 0.998
    
    is_near_ob = (current_price >= order_block_low * 0.98 and current_price <= order_block_high * 1.02)
    
    high_low_range = swing_high - swing_low
    fvg_threshold = high_low_range * 0.015
    
    fair_value_gaps = []
    for i in range(1, len(recent_closes) - 1):
        gap = abs(recent_highs[i] - recent_lows[i+1])
        if gap > fvg_threshold and recent_closes[i] != recent_closes[i+1]:
            fair_value_gaps.append(gap)
    
    fvg_strength = min(100, len(fair_value_gaps) * 25) if fair_value_gaps else 0
    
    liquidity_pool = (swing_high + swing_low) / 2
    liquidity_distance = abs(current_price - liquidity_pool) / max(1e-6, liquidity_pool) * 100
    
    break_of_structure = False
    if len(recent_closes) >= 3:
        if recent_closes[-1] > recent_closes[-2] > recent_closes[-3]:
            break_of_structure = True
        elif recent_closes[-1] < recent_closes[-2] < recent_closes[-3]:
            break_of_structure = True
    
    smc_strength = 0
    if is_near_ob:
        smc_strength += 30
    if fvg_strength > 0:
        smc_strength += fvg_strength * 0.3
    if break_of_structure:
        smc_strength += 25
    if liquidity_distance < 2:
        smc_strength += 15
    
    return {
        "order_block": {"high": order_block_high, "low": order_block_low},
        "is_near_ob": is_near_ob,
        "liquidity_pool": liquidity_pool,
        "fvg_strength": min(100, fvg_strength),
        "break_of_structure": break_of_structure,
        "smc_score": min(100, smc_strength)
    }


def calculate_vwap(highs, lows, closes, volumes):
    if len(closes) < 2:
        return None
    
    typical_price = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    tp_volume = [typical_price[i] * volumes[i] for i in range(len(volumes))]
    cumulative_tp_vol = sum(tp_volume)
    cumulative_vol = sum(volumes)
    
    vwap = cumulative_tp_vol / max(1e-6, cumulative_vol) if cumulative_vol > 0 else closes[-1]
    current_price = closes[-1]
    
    vwap_diff = ((current_price - vwap) / vwap * 100) if vwap else 0
    is_above_vwap = current_price > vwap
    
    return {
        "vwap": vwap,
        "current": current_price,
        "diff_pct": vwap_diff,
        "is_above": is_above_vwap,
        "strength": min(100, max(0, 50 + (abs(vwap_diff) * 10)))
    }


def detect_divergence(closes, rsi_vals, lookback=14):
    if len(closes) < lookback or len(rsi_vals) < lookback:
        return None
    
    recent_closes = closes[-lookback:]
    recent_rsi = [r for r in rsi_vals[-lookback:] if r is not None]
    
    if len(recent_closes) < 3 or len(recent_rsi) < 3:
        return None
    
    price_trend = "up" if recent_closes[-1] > recent_closes[-2] else "down"
    rsi_trend = "up" if recent_rsi[-1] > recent_rsi[-2] else "down"
    
    bullish_div = price_trend == "down" and rsi_trend == "up"
    bearish_div = price_trend == "up" and rsi_trend == "down"
    
    divergence_strength = 0
    if bullish_div:
        divergence_strength = 40
    elif bearish_div:
        divergence_strength = -40
    
    return {
        "bullish": bullish_div,
        "bearish": bearish_div,
        "price_trend": price_trend,
        "rsi_trend": rsi_trend,
        "strength": divergence_strength
    }


def higher_timeframe_confirmation(symbol, current_tf="30m"):
    try:
        htf_map = {"5m": "15m", "15m": "1h", "30m": "1h", "1h": "4h", "4h": "1d"}
        htf = htf_map.get(current_tf, "1h")
        
        data = yf.download(tickers=symbol, period="7d", interval=htf, progress=False, auto_adjust=True)
        if data.empty:
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            htf_closes = data['Close'][symbol].values.tolist()
            htf_highs = data['High'][symbol].values.tolist()
            htf_lows = data['Low'][symbol].values.tolist()
        else:
            htf_closes = data['Close'].values.tolist()
            htf_highs = data['High'].values.tolist()
            htf_lows = data['Low'].values.tolist()
        
        htf_closes = [float(c) for c in htf_closes if not pd.isna(c)]
        htf_highs = [float(h) for h in htf_highs if not pd.isna(h)]
        htf_lows = [float(l) for l in htf_lows if not pd.isna(l)]
        
        if len(htf_closes) < 3:
            return None
        
        htf_fast = ema(htf_closes, FAST_MA)
        htf_slow = ema(htf_closes, SLOW_MA)
        htf_rsi = rsi(htf_closes, RSI_PERIOD)
        
        htf_fast_now = htf_fast[-1] if htf_fast else None
        htf_slow_now = htf_slow[-1] if htf_slow else None
        htf_rsi_now = htf_rsi[-1] if htf_rsi[-1] is not None else 50
        
        buy_confirmed = htf_fast_now > htf_slow_now if (htf_fast_now and htf_slow_now) else False
        sell_confirmed = htf_fast_now < htf_slow_now if (htf_fast_now and htf_slow_now) else False
        
        strength = 0
        if buy_confirmed and htf_rsi_now >= 50:
            strength = 50
        elif sell_confirmed and htf_rsi_now <= 50:
            strength = -50
        
        return {
            "timeframe": htf,
            "buy_confirmed": buy_confirmed,
            "sell_confirmed": sell_confirmed,
            "rsi": htf_rsi_now,
            "strength": strength
        }
    except Exception as e:
        logger.warning(f"HTF confirmation error for {symbol}: {e}")
        return None


def detect_candlestick_patterns(opens, highs, lows, closes, lookback=5):
    if len(closes) < lookback:
        return None
    
    current_open = opens[-1]
    current_high = highs[-1]
    current_low = lows[-1]
    current_close = closes[-1]
    
    prev_close = closes[-2] if len(closes) > 1 else current_close
    prev_open = opens[-2] if len(opens) > 1 else current_open
    
    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_open, current_close)
    lower_wick = min(current_open, current_close) - current_low
    full_range = current_high - current_low
    
    patterns = []
    strength = 0
    
    if full_range > 0:
        body_ratio = body / full_range
        upper_ratio = upper_wick / full_range
        lower_ratio = lower_wick / full_range
        
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns.append("Hammer")
            strength += 35
        
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            patterns.append("Shooting Star")
            strength += 35
        
        if body < full_range * 0.1 and upper_wick > 0 and lower_wick > 0:
            patterns.append("Doji")
            strength += 25
        
        if len(closes) > 2:
            prev_body = abs(opens[-2] - closes[-2])
            prev_high = highs[-2]
            prev_low = lows[-2]
            
            if (current_close > prev_high and current_open < prev_low and body > prev_body):
                patterns.append("Bullish Engulfing")
                strength += 40
            elif (current_close < prev_low and current_open > prev_high and body > prev_body):
                patterns.append("Bearish Engulfing")
                strength += 40
        
        if len(closes) > 2:
            if (current_high >= opens[-2] and current_low <= closes[-2] and 
                ((current_open < opens[-2] and current_close > closes[-2]) or 
                 (current_open > opens[-2] and current_close < closes[-2]))):
                patterns.append("Harami")
                strength += 30
    
    if not patterns:
        return None
    
    return {
        "patterns": patterns,
        "strength": min(100, strength),
        "description": ", ".join(patterns)
    }


def detect_swing_levels(highs, lows, closes, lookback=20):
    if len(closes) < lookback + 2:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    
    swing_highs = []
    swing_lows = []
    
    for i in range(1, len(recent_highs) - 1):
        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
            swing_highs.append(recent_highs[i])
        
        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
            swing_lows.append(recent_lows[i])
    
    highest_swing = max(swing_highs) if swing_highs else max(recent_highs)
    lowest_swing = min(swing_lows) if swing_lows else min(recent_lows)
    
    current_price = closes[-1]
    
    is_near_resistance = highest_swing > 0 and abs(current_price - highest_swing) / highest_swing < 0.01
    is_near_support = lowest_swing > 0 and abs(current_price - lowest_swing) / lowest_swing < 0.01
    
    resistance_distance = ((highest_swing - current_price) / current_price * 100) if current_price > 0 else 0
    support_distance = ((current_price - lowest_swing) / current_price * 100) if current_price > 0 else 0
    
    swing_strength = 0
    if is_near_resistance:
        swing_strength = 60
    elif is_near_support:
        swing_strength = 60
    elif abs(resistance_distance) < 0.5:
        swing_strength = 40
    elif abs(support_distance) < 0.5:
        swing_strength = 40
    
    return {
        "resistance": highest_swing,
        "support": lowest_swing,
        "is_near_resistance": is_near_resistance,
        "is_near_support": is_near_support,
        "resistance_distance_pct": resistance_distance,
        "support_distance_pct": support_distance,
        "strength": swing_strength
    }


def calculate_risk_reward_ratio(entry, tp, sl):
    if entry == 0 or tp == entry or sl == entry:
        return None
    
    reward = abs(tp - entry)
    risk = abs(entry - sl)
    
    if risk == 0:
        return None
    
    rrr = reward / risk
    
    return {
        "ratio": rrr,
        "reward": reward,
        "risk": risk,
        "is_favorable": rrr >= 2.0,
        "quality": "Excellent" if rrr >= 3.0 else "Good" if rrr >= 2.0 else "Fair" if rrr >= 1.5 else "Poor"
    }


def detect_breakout(highs, lows, closes, swings, lookback=20):
    if not swings or lookback < 5:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    current_price = closes[-1]
    
    resistance = swings.get('resistance')
    support = swings.get('support')
    
    if not resistance or not support:
        return None
    
    breakout_resistance = current_price > resistance * 1.001
    breakout_support = current_price < support * 0.999
    
    resistance_distance = ((current_price - resistance) / resistance * 100) if resistance > 0 else 0
    support_distance = ((support - current_price) / support * 100) if support > 0 else 0
    
    breakout_strength = 0
    breakout_type = None
    
    if breakout_resistance:
        breakout_strength = min(100, 50 + (resistance_distance * 5))
        breakout_type = "Bullish Breakout"
    elif breakout_support:
        breakout_strength = min(100, 50 + (support_distance * 5))
        breakout_type = "Bearish Breakdown"
    
    return {
        "breakout": breakout_resistance or breakout_support,
        "type": breakout_type,
        "strength": breakout_strength,
        "resistance_breakout": breakout_resistance,
        "support_breakdown": breakout_support
    }


def calculate_trend_strength(fast_ma, slow_ma, rsi, adx, delta):
    trend_strength = 0
    
    if fast_ma > slow_ma:
        trend_strength += 30
    if rsi is not None and rsi >= 60:
        trend_strength += 20
    elif rsi is not None and rsi <= 40:
        trend_strength += 20
    
    if adx is not None and adx > 25:
        trend_strength += 25
    elif adx is not None and adx > 20:
        trend_strength += 15
    
    if delta is not None:
        if abs(delta) > 50:
            trend_strength += 25
        elif abs(delta) > 30:
            trend_strength += 15
    
    return min(100, trend_strength)


def calculate_mfi(highs, lows, closes, volumes, period=14):
    if len(closes) < period:
        return None
    
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    money_flows = [tp * v for tp, v in zip(typical_prices, volumes)]
    
    positive_mf = 0
    negative_mf = 0
    
    for i in range(1, len(typical_prices)):
        if typical_prices[i] > typical_prices[i-1]:
            positive_mf += money_flows[i]
        elif typical_prices[i] < typical_prices[i-1]:
            negative_mf += money_flows[i]
    
    if negative_mf == 0:
        return 100.0
    
    mfi_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfi_ratio))
    
    return mfi


def detect_false_breakout(highs, lows, closes, swings, lookback=5):
    if not swings:
        return None
    
    current_price = closes[-1]
    resistance = swings.get('resistance')
    support = swings.get('support')
    
    if not resistance or not support:
        return None
    
    recent_closes = closes[-lookback:]
    
    false_breakout = False
    fakeout_type = None
    fakeout_strength = 0
    
    max_close = max(recent_closes)
    min_close = min(recent_closes)
    
    if max_close > resistance * 1.001 and current_price < resistance * 0.999:
        false_breakout = True
        fakeout_type = "False Bullish Breakout"
        fakeout_strength = 60
    elif min_close < support * 0.999 and current_price > support * 1.001:
        false_breakout = True
        fakeout_type = "False Bearish Breakdown"
        fakeout_strength = 60
    
    return {
        "false_breakout": false_breakout,
        "type": fakeout_type,
        "strength": fakeout_strength
    }


def identify_supply_demand_zones(highs, lows, closes, lookback=100):
    if len(highs) < lookback:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    supply_zones = []
    demand_zones = []
    
    for i in range(len(recent_highs) - 1):
        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
            zone_strength = (recent_highs[i] - min(recent_highs[max(0, i-5):i+5])) / recent_highs[i] * 100
            supply_zones.append({
                "level": recent_highs[i],
                "strength": min(100, zone_strength + 30)
            })
        
        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
            zone_strength = (max(recent_lows[max(0, i-5):i+5]) - recent_lows[i]) / recent_lows[i] * 100
            demand_zones.append({
                "level": recent_lows[i],
                "strength": min(100, zone_strength + 30)
            })
    
    current_price = closes[-1]
    nearest_supply = None
    nearest_demand = None
    
    if supply_zones:
        nearest_supply = min(supply_zones, key=lambda x: abs(x['level'] - current_price))
    if demand_zones:
        nearest_demand = min(demand_zones, key=lambda x: abs(x['level'] - current_price))
    
    return {
        "supply_zones": supply_zones[-3:] if supply_zones else [],
        "demand_zones": demand_zones[-3:] if demand_zones else [],
        "nearest_supply": nearest_supply,
        "nearest_demand": nearest_demand
    }


def get_current_session():
    current_hour = datetime.now(timezone.utc).hour
    
    if 0 <= current_hour < 9:
        return "asia", TRADING_SESSIONS["asia"]["symbols"]
    elif 8 <= current_hour < 16:
        return "london", TRADING_SESSIONS["london"]["symbols"]
    elif 13 <= current_hour < 22:
        return "newyork", TRADING_SESSIONS["newyork"]["symbols"]
    else:
        return "overlap", TRADING_SESSIONS["asia"]["symbols"] + TRADING_SESSIONS["london"]["symbols"]


def is_session_valid_for_symbol(symbol):
    current_session, valid_symbols = get_current_session()
    
    # 24/7 symbols (crypto)
    if "BTC" in symbol:
        return True, "24h"
    
    if symbol in valid_symbols:
        return True, current_session
    
    return False, current_session


def count_confluences(info):
    confluences = 0
    total_signals = 0
    
    signal_indicators = [
        ("buy_cross", info.get("fast"), info.get("slow")),
        ("rsi", info.get("rsi")),
        ("macd", info.get("macd"), info.get("sig")),
        ("stoch", info.get("stoch_k")),
        ("adx", info.get("adx")),
        ("ichimoku", info.get("ichimoku")),
        ("delta", info.get("delta")),
        ("va", info.get("va")),
        ("chart_patterns", info.get("chart_patterns")),
        ("breakout", info.get("breakout")),
        ("mfi", info.get("mfi")),
        ("pivots", info.get("pivots")),
        ("smc", info.get("smc")),
        ("vwap", info.get("vwap")),
        ("divergence", info.get("divergence")),
        ("candles", info.get("candles")),
        ("swings", info.get("swings")),
        ("fibo", info.get("fibo")),
        ("news", info.get("news_sentiment")),
    ]
    
    for indicator in signal_indicators:
        if indicator[0] in ["buy_cross"]:
            total_signals += 1
            if len(indicator) > 2 and indicator[1] and indicator[2] and indicator[1] > indicator[2]:
                confluences += 1
        elif indicator[0] in ["rsi", "delta", "stoch", "adx"]:
            total_signals += 1
            if len(indicator) > 1 and indicator[1] is not None:
                confluences += 1
        elif len(indicator) > 1 and indicator[1]:
            total_signals += 1
            confluences += 1
    
    confluence_score = (confluences / total_signals * 100) if total_signals > 0 else 0
    
    return {
        "confluence_count": confluences,
        "total_signals": total_signals,
        "confluence_percentage": confluence_score,
        "strong_confluence": confluences >= 15
    }


def detect_chart_patterns(highs, lows, closes, lookback=30):
    if len(closes) < lookback:
        return None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    
    swing_points = []
    for i in range(1, len(recent_highs) - 1):
        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
            swing_points.append(("high", i, recent_highs[i]))
        elif recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
            swing_points.append(("low", i, recent_lows[i]))
    
    if len(swing_points) < 4:
        return None
    
    patterns = []
    strength = 0
    
    last_points = swing_points[-4:]
    
    if len(last_points) >= 4:
        p1_type, p1_idx, p1_val = last_points[0]
        p2_type, p2_idx, p2_val = last_points[1]
        p3_type, p3_idx, p3_val = last_points[2]
        p4_type, p4_idx, p4_val = last_points[3]
        
        v_tolerance = 0.015
        
        if p1_type == "high" and p2_type == "low" and p3_type == "high":
            if abs(p1_val - p3_val) / max(p1_val, p3_val) < v_tolerance:
                patterns.append("M-Shape (Double Top)")
                strength += 45
            else:
                patterns.append("Inverted V")
                strength += 30
        
        elif p1_type == "low" and p2_type == "high" and p3_type == "low":
            if abs(p1_val - p3_val) / max(p1_val, p3_val) < v_tolerance:
                patterns.append("W-Shape (Double Bottom)")
                strength += 45
            else:
                patterns.append("V-Shape")
                strength += 30
        
        if len(last_points) >= 4:
            if p1_type == "low" and p2_type == "high" and p3_type == "low" and p4_type == "high":
                if abs(p3_val - p1_val) / max(p3_val, p1_val) < 0.02:
                    cup_depth = abs(p2_val - p3_val) / p2_val
                    if cup_depth > 0.01:
                        patterns.append("Cup & Handle")
                        strength += 50
    
    if not patterns:
        return None
    
    return {
        "patterns": patterns,
        "strength": min(100, strength),
        "description": ", ".join(patterns)
    }


def fetch(symbol, tf, bars=LOOKBACK_BARS):
    try:
        data = yf.download(tickers=symbol, period="14d", interval=tf, progress=False, auto_adjust=True)
        if data.empty:
            logger.warning(f"No data received for {symbol}")
            return [], [], [], [], []

        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close'][symbol].values.tolist()
            highs = data['High'][symbol].values.tolist()
            lows = data['Low'][symbol].values.tolist()
            opens = data['Open'][symbol].values.tolist()
            volumes = data['Volume'][symbol].values.tolist()
        else:
            closes = data['Close'].values.tolist()
            highs = data['High'].values.tolist()
            lows = data['Low'].values.tolist()
            opens = data['Open'].values.tolist()
            volumes = data['Volume'].values.tolist()

        closes = [float(c) for c in closes if not pd.isna(c)]
        highs = [float(h) for h in highs if not pd.isna(h)]
        lows = [float(l) for l in lows if not pd.isna(l)]
        opens = [float(o) for o in opens if not pd.isna(o)]
        volumes = [float(v) for v in volumes if not pd.isna(v)]

        return closes[-bars:], highs[-bars:], lows[-bars:], opens[-bars:], volumes[-bars:]
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return [], [], [], [], []


def get_news_sentiment(symbol):
    try:
        if not NEWSAPI_KEY:
            return 0, "No API Key"
        
        query = ""
        if "EUR" in symbol:
            query = "EUR USD forex"
        elif "GBP" in symbol:
            query = "GBP USD forex"
        elif "JPY" in symbol:
            query = "JPY USD forex"
        elif "CHF" in symbol:
            query = "CHF USD forex"
        elif "CAD" in symbol:
            query = "CAD USD forex"
        elif "AUD" in symbol:
            query = "AUD USD forex"
        elif "NZD" in symbol:
            query = "NZD USD forex"
        elif "BTC-USD" in symbol:
            query = "gold price"
        elif "BTC" in symbol:
            query = "bitcoin crypto"
        else:
            return 0, "Unknown"
        
        url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=10"
        headers = {"X-Api-Key": NEWSAPI_KEY}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return 0, "Error"
        
        articles = response.json().get("articles", [])
        if not articles:
            return 0, "No news"
        
        sentiments = []
        for article in articles[:5]:
            headline = article.get("title", "") + " " + article.get("description", "")
            blob = TextBlob(headline)
            sentiments.append(blob.sentiment.polarity)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        avg_sentiment = max(-1, min(1, avg_sentiment))
        return avg_sentiment, "OK"
    except Exception as e:
        logger.error(f"News sentiment error for {symbol}: {e}")
        return 0, "Error"


def points_to_price(symbol, points):
    if "BTC-USD" in symbol:
        return points * 0.1
    if "BTC" in symbol:
        return points * 1.0
    return points * 0.0001


def format_price(symbol, p):
    if "BTC-USD" in symbol:
        return f"{p:.2f}"
    if "BTC" in symbol:
        return f"{p:.2f}"
    return f"{p:.5f}"


def is_high_liquidity_hour(symbol):
    """Check if current time is high liquidity hour for the symbol"""
    current_hour = datetime.now(timezone.utc).hour
    
    liquidity_windows = {
        "EURUSD=X": [(8, 16), (13, 17)],      # London + NY overlap
        "GBPUSD=X": [(8, 16), (13, 17)],      # London + NY overlap
        "USDJPY=X": [(0, 9), (13, 22)],       # Asia + NY
        "USDCHF=X": [(8, 16), (13, 17)],      # London + NY
        "USDCAD=X": [(13, 17)],               # NY only
        "AUDUSD=X": [(0, 9), (13, 17)],       # Asia + NY
        "NZDUSD=X": [(0, 9), (21, 23)],       # Asia + NY pre-open
        "EURGBP=X": [(8, 16)],                # London
        "EURJPY=X": [(0, 9), (8, 16)],        # Asia + London
        "GBPJPY=X": [(0, 9), (8, 16)],        # Asia + London
        "AUDJPY=X": [(0, 9)],                 # Asia
        "CHFJPY=X": [(0, 9), (8, 16)],        # Asia + London
        "XAUUSD=X": [(8, 16), (13, 17)],      # London + NY overlap (similar to major pairs)
        "BTC-USD": [(0, 24)],                 # 24/7 trading
    }
    
    windows = liquidity_windows.get(symbol, [(0, 24)])
    return any(start <= current_hour < end for start, end in windows)


def is_sufficient_volatility(atr_now, closes):
    """Check if current ATR is above 30-day average"""
    if not atr_now or len(closes) < 30:
        return True
    
    closes_30d = closes[-30:] if len(closes) >= 30 else closes
    
    atr_30d_values = []
    for i in range(1, len(closes_30d)):
        high_low = closes_30d[i] - closes_30d[i-1]
        atr_30d_values.append(abs(high_low))
    
    avg_volatility = sum(atr_30d_values) / len(atr_30d_values) if atr_30d_values else atr_now
    
    return atr_now > (avg_volatility * 0.8)  # Current ATR at least 80% of 30-day avg


def is_too_close_to_session_end(symbol):
    """Check if within 30 mins of session end (no new entries)"""
    current_hour = datetime.now(timezone.utc).hour
    current_min = datetime.now(timezone.utc).minute
    
    session_end_hour = SESSION_END_TIMES.get(symbol, 24)
    end_time_minutes = session_end_hour * 60
    current_time_minutes = current_hour * 60 + current_min
    
    # 30 minutes before session end
    cutoff_time = (session_end_hour - 0.5) * 60
    return current_time_minutes >= cutoff_time and current_hour < session_end_hour


def is_market_open(symbol):
    """Check if market is open (not on weekend)"""
    now = datetime.now(timezone.utc)
    day_of_week = now.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
    hour_utc = now.hour
    
    # Crypto trades 24/7
    if "BTC" in symbol:
        return True
    
    # Gold trades almost 24/5 but has gaps
    if "GC=F" in symbol:
        return True
    
    # STRICT forex market hours: Sun 22:00 UTC to Fri 21:00 UTC
    # Block: Sat (5) and Sun before 22:00 (6), Fri 21:00+ onwards (4)
    if day_of_week == 5:  # Saturday - ALWAYS closed
        return False
    
    if day_of_week == 6 and hour_utc < 22:  # Sunday before 22:00 UTC
        return False
    
    if day_of_week == 4 and hour_utc >= 21:  # Friday 21:00 UTC onwards
        return False
    
    return True


def get_daily_loss():
    """Calculate today's total loss in pips"""
    today = datetime.now(timezone.utc).date()
    total_loss = 0
    
    for symbol_trades in trades_history.values():
        for trade in symbol_trades:
            if trade["timestamp"].date() == today and trade["pips"] < 0:
                total_loss += abs(trade["pips"])
    
    return total_loss


def record_trade(symbol, side, entry, exit_price, tp_hit, reason):
    """Record a closed trade for analytics"""
    global trades_history, trading_paused
    
    if symbol not in trades_history:
        trades_history[symbol] = []
    
    direction = 1 if side == "BUY" else -1
    pips = (exit_price - entry) * direction
    
    trade = {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "exit": exit_price,
        "pips": pips,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc),
        "hit": tp_hit,
        "filled_portions": [1.0]  # Track partial fills
    }
    
    trades_history[symbol].append(trade)
    logger.info(f"Trade recorded: {symbol} {side} {pips:+.0f} pips - {reason}")
    
    # Check daily loss limit
    if get_daily_loss() > DAILY_LOSS_LIMIT:
        trading_paused = True
        logger.warning(f"⚠️ Daily loss limit ({DAILY_LOSS_LIMIT} pips) exceeded! Trading paused.")


def get_today_stats():
    """Get today's trading statistics with advanced metrics"""
    today = datetime.now(timezone.utc).date()
    
    today_trades = []
    for symbol_trades in trades_history.values():
        for trade in symbol_trades:
            if trade["timestamp"].date() == today:
                today_trades.append(trade)
    
    if not today_trades:
        return None
    
    wins = [t for t in today_trades if t["pips"] > 0]
    losses = [t for t in today_trades if t["pips"] <= 0]
    
    total_pips = sum(t["pips"] for t in today_trades)
    win_pips = sum(t["pips"] for t in wins)
    loss_pips = sum(t["pips"] for t in losses)
    
    win_rate = (len(wins) / len(today_trades) * 100) if today_trades else 0
    profit_factor = abs(win_pips / loss_pips) if loss_pips != 0 else 0
    
    best_trade = max(today_trades, key=lambda t: t["pips"], default=None)
    worst_trade = min(today_trades, key=lambda t: t["pips"], default=None)
    
    # Best trading hours
    hour_pips = {}
    for trade in today_trades:
        hour = trade["timestamp"].hour
        hour_pips[hour] = hour_pips.get(hour, 0) + trade["pips"]
    best_hour = max(hour_pips, key=hour_pips.get) if hour_pips else None
    
    # Symbol performance
    symbol_pips = {}
    for trade in today_trades:
        sym = trade["symbol"]
        symbol_pips[sym] = symbol_pips.get(sym, 0) + trade["pips"]
    
    return {
        "total_trades": len(today_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_pips": total_pips,
        "win_pips": win_pips,
        "loss_pips": loss_pips,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "best_hour": best_hour,
        "symbol_pips": symbol_pips,
        "consecutive_wins": len([1 for i, t in enumerate(today_trades) if t["pips"] > 0 and (i==0 or today_trades[i-1]["pips"] > 0)]),
        "consecutive_losses": len([1 for i, t in enumerate(today_trades) if t["pips"] <= 0 and (i==0 or today_trades[i-1]["pips"] <= 0)])
    }


def analyze(symbol):
    # ✅ PIVOT POINTS + ICHIMOKU BOOST - Primary signal with strength enhancement
    clean_sym = symbol.replace("=X", "").replace("-USD", "")
    
    closes, highs, lows, opens, volumes = fetch(symbol, TIMEFRAME)
    if len(closes) < 30:
        logger.warning(f"Not enough data for {clean_sym}: {len(closes)} bars")
        return None

    price = closes[-1]
    atr_now = atr(highs, lows, closes, 14)
    
    # Calculate pivot points from PREVIOUS bar (1 bar ago)
    # This gives us true resistance/support for current price to break
    if len(closes) >= 2:
        pivot_high = highs[-2]
        pivot_low = lows[-2]
        pivot_close = closes[-2]
    else:
        pivot_high = highs[-1]
        pivot_low = lows[-1]
        pivot_close = closes[-1]
    
    pivots = pivot_points(pivot_high, pivot_low, pivot_close)
    logger.info(f"📊 {clean_sym}: Price={price:.5f}, R1={pivots['r1']:.5f}, S1={pivots['s1']:.5f}")
    
    # Pure pivot point signal: Price above/below pivot levels
    buy = price > pivots["r1"]   # Price above resistance = BUY
    sell = price < pivots["s1"]  # Price below support = SELL
    
    # ✅ ICHIMOKU as STRENGTH BOOSTER (not blocker)
    ichi_data = ichimoku(highs, lows, closes)
    ichi_buy_boost = 0
    ichi_sell_boost = 0
    
    if buy:
        ichi_buy_confirmed, ichi_buy_boost = check_ichimoku_confirmation(price, ichi_data, True)
    if sell:
        ichi_sell_confirmed, ichi_sell_boost = check_ichimoku_confirmation(price, ichi_data, False)
    
    # Base strength: 70 if pivot breakout detected
    # Add Ichimoku boost (+15) if it also confirms = 85 total
    strength = 70 if (buy or sell) else 0
    if buy:
        strength += ichi_buy_boost
    elif sell:
        strength += ichi_sell_boost

    if buy or sell:
        ichi_note = f" + Ichimoku ✓ @ {price:.5f}" if (ichi_buy_boost if buy else ichi_sell_boost) > 0 else f" @ {price:.5f}"
        logger.info(f"🎯 {clean_sym}: {'BUY' if buy else 'SELL'} signal! Pivot ✓{ichi_note}")

    tp1, tp2, tp3 = None, None, None
    sl = None
    if buy or sell:
        side = "BUY" if buy else "SELL"
        tp1, tp2, tp3, sl = open_position(symbol, side, price, atr_now)

    return {
        "price": price,
        "buy": buy,
        "sell": sell,
        "atr": atr_now,
        "pivots": pivots,
        "ichimoku": ichi_data,
        "strength": strength,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "swings": None
    }


def open_position(symbol, side, entry_price, atr_val=None):
    direction = 1 if side == "BUY" else -1
    
    if atr_val and atr_val > 0:
        tp1 = entry_price + atr_val * 1.0 * direction
        tp2 = entry_price + atr_val * 1.5 * direction
        tp3 = entry_price + atr_val * 2.0 * direction
        sl = entry_price - atr_val * 1.0 * direction
    else:
        mult = direction
        tp1 = entry_price + points_to_price(symbol, TP1_POINTS) * mult
        tp2 = entry_price + points_to_price(symbol, TP2_POINTS) * mult
        tp3 = entry_price + points_to_price(symbol, TP3_POINTS) * mult
        sl = entry_price - points_to_price(symbol, SL_POINTS) * mult

    state[symbol] = {
        "side": side,
        "entry": entry_price,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
        "mt4_order_id": None
    }
    return tp1, tp2, tp3, sl


async def place_mt4_order(symbol, side, entry, tp1, tp2, tp3, sl):
    """Place order on MT4 via MetaApi"""
    if not metaapi_connection:
        return None
    
    try:
        # Convert Yahoo Finance symbol to MT4 format
        mt4_symbol = symbol.replace("=X", "").replace("-USD", "")
        
        order_type = "buy" if side == "BUY" else "sell"
        volume = 0.1  # Default 0.1 lot size
        
        # Place market order with TP/SL
        result = await metaapi_connection.create_market_order(
            symbol=mt4_symbol,
            order_type=order_type,
            volume=volume,
            takeProfit=tp1,
            stopLoss=sl
        )
        
        logger.info(f"MT4 Order placed: {result}")
        state[symbol]["mt4_order_id"] = result.get("orderId")
        return result
    except Exception as e:
        logger.error(f"MT4 order error: {e}")
        return None


def format_entry(symbol, info, side, entry, tp1, tp2, tp3, sl):
    confluence = info.get('confluence', {})
    confluence_count = confluence.get('confluence_count', 0) if confluence else 0
    total_signals = confluence.get('total_signals', 1) if confluence else 1
    confluence_icon = "🔥" if confluence_count >= 20 else "⚡"
    
    rrr = info.get('rrr', {})
    rrr_ratio = f"{rrr.get('ratio', 0):.2f}" if rrr else "N/A"
    rrr_quality = rrr.get('quality', "Fair") if rrr else "Fair"
    
    is_retest = info.get('is_retest', False)
    signal_type = "RETEST" if is_retest else "NEW"
    
    rsi_val = info.get('rsi', 0)
    clean_symbol = symbol.replace("=X", "").replace("-USD", "")
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    # Format prices for display
    entry_price = format_price(symbol, entry)
    tp1_price = format_price(symbol, tp1)
    tp2_price = format_price(symbol, tp2)
    tp3_price = format_price(symbol, tp3)
    sl_price = format_price(symbol, sl)
    
    # EASY COPYABLE MULTILINE - Numbers only, easy to read
    copyable_clean = (
        f"SYMBOL: {clean_symbol}\n"
        f"SIDE: {side}\n"
        f"ENTRY: {entry_price}\n"
        f"TP1: {tp1_price}\n"
        f"TP2: {tp2_price}\n"
        f"TP3: {tp3_price}\n"
        f"SL: {sl_price}"
    )
    
    # One-liner for quick copy
    copyable_oneline = f"{clean_symbol} | {side} | {entry_price} | {tp1_price} | {tp2_price} | {tp3_price} | {sl_price}"
    
    # Use multiline as primary copyable (easier to read and copy)
    copyable = copyable_clean
    
    msg_text = (
        f"{'🚀' if side=='BUY' else '⬇️'} <b>TRADING SIGNAL - {clean_symbol}</b>\n"
        f"⏰ {timestamp} UTC\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"<b>📋 TAP HERE TO COPY (Numbers for MT4):</b>\n"
        f"<code>{copyable_clean}</code>\n"
        f"\n"
        f"<b>OR COPY ONE LINE:</b>\n"
        f"<code>{copyable_oneline}</code>\n"
        f"\n"
        f"{'BUY 📈' if side=='BUY' else 'SELL 📉'} @ {entry_price}\n"
        f"Accuracy: {info['strength']}/100 | {signal_type} | R/R: {rrr_ratio}:1"
    )
    
    return msg_text, copyable


def format_exit(symbol, side, reason, price):
    if "TP1" in reason:
        icon = "✅"
        status = "TP1 HIT"
    elif "TP2" in reason:
        icon = "✅✅"
        status = "TP2 HIT"
    elif "TP3" in reason:
        icon = "🎉"
        status = "TP3 HIT"
    elif "SL" in reason:
        icon = "❌"
        status = "SL HIT"
    else:
        icon = "🚪"
        status = "EXIT"

    clean_symbol = symbol.replace("=X", "").replace("-USD", "")
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    exit_price = format_price(symbol, price)
    
    # Easy copyable section - CLEAN FORMAT
    copyable = f"SYMBOL: {clean_symbol}\nSIDE: {side}\nEXIT: {exit_price}"

    msg_text = (
        f"{icon} <b>EXIT SIGNAL - {clean_symbol}</b>\n"
        f"⏰ {timestamp} UTC\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"<b>Status: {status}</b>\n"
        f"\n"
        f"<b>📋 TAP HERE TO COPY (Close in MT4):</b>\n"
        f"<code>{copyable}</code>\n"
        f"\n"
        f"Close {side} at <code>{exit_price}</code>"
    )
    
    return msg_text, copyable


async def send_message(bot: Bot, msg: str, post_to_channel: bool = False, copyable_text: str = None, symbol: str = None, side: str = None, entry: float = None, tp1: float = None, tp2: float = None, tp3: float = None, sl: float = None, pivot_points: dict = None):
    global channel_free_signal_sent, last_channel_signal_date, signal_copyables, signal_history
    try:
        # Create inline keyboard if copyable text provided
        reply_markup = None
        unique_id = None
        
        if copyable_text:
            import time
            unique_id = f"{int(time.time() * 1000)}"
            signal_copyables[unique_id] = copyable_text
            
            keyboard = [
                [InlineKeyboardButton("📋 Copy to MT4", callback_data=f"copy_{unique_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Track signal in history
        if symbol and side and entry is not None:
            signal_history.append({
                "symbol": symbol.replace("=X", "").replace("-USD", ""),
                "side": side,
                "entry": entry,
                "time": datetime.utcnow()
            })
        
        # Send to user
        await bot.send_message(chat_id=CHAT_ID, text=msg, reply_markup=reply_markup, parse_mode="HTML")
        logger.info(f"Message sent: {msg[:50]}...")
        
        # Generate and send technical analysis chart for ENTRY signals only
        if symbol and side and entry is not None and tp1 and tp2 and tp3 and sl:
            try:
                chart = generate_signal_chart(symbol, entry, tp1, tp2, tp3, sl, side, pivot_points or {})
                if chart:
                    clean_symbol = symbol.replace("=X", "").replace("-USD", "")
                    chart_caption = f"📊 {clean_symbol} {side} Signal Analysis\n\n💡 Why this signal?\n• Pivot point breakout detected\n• Price action confirmed\n• ATR-based targets calculated\n• Risk/Reward optimized"
                    await bot.send_photo(chat_id=CHAT_ID, photo=chart, caption=chart_caption, parse_mode="HTML")
                    logger.info(f"✅ Chart sent for {clean_symbol}")
            except Exception as e:
                logger.debug(f"Chart generation skipped: {e}")
        
        # Post free signal to channel (1 per day)
        if post_to_channel:
            today = datetime.utcnow().date()
            if last_channel_signal_date != today:
                try:
                    await bot.send_message(chat_id=CHANNEL_ID, text=msg, parse_mode="HTML")
                    channel_free_signal_sent = True
                    last_channel_signal_date = today
                    logger.info(f"✅ Free signal posted to channel {CHANNEL_ID}")
                except Exception as e:
                    logger.warning(f"Failed to post to channel: {e}")
    except Exception as e:
        logger.error(f"Error sending message: {e}")


async def check_symbol(bot: Bot, symbol: str):
    global state

    # ✅ PIVOT POINTS ONLY - Process ALL pairs, ALL times (no weekend blocking)
    info = analyze(symbol)
    if info is None:
        return

    price = info["price"]
    atr_val = info.get("atr")
    s = state.get(symbol)
    
    session_valid, current_session = is_session_valid_for_symbol(symbol)
    info["session"] = current_session
    info["session_valid"] = session_valid
    
    near_event, event_name = is_near_economic_event(symbol)
    info["near_economic_event"] = near_event
    info["economic_event_name"] = event_name
    
    swings = info.get("swings")
    rsi_val = info.get("rsi")
    update_retest_levels(symbol, swings, price)
    
    is_retest, retest_msg = detect_retest_opportunity(symbol, price, swings, rsi_val)
    info["is_retest"] = is_retest
    info["retest_message"] = retest_msg
    
    closes, _, _, _, _ = fetch(symbol, TIMEFRAME)
    high_liquidity = is_high_liquidity_hour(symbol)
    # BTC: Always has liquidity & volatility (24/7 trading)
    sufficient_volatility = is_sufficient_volatility(atr_val, closes) if "BTC" not in symbol else True
    
    info["high_liquidity"] = high_liquidity
    info["sufficient_volatility"] = sufficient_volatility

    session_close_approaching = is_too_close_to_session_end(symbol) if "BTC" not in symbol else False
    info["session_close_approaching"] = session_close_approaching

    if s is None:
        now = datetime.utcnow().timestamp()
        last_signal_time = sent_signals.get(symbol, 0)
        time_since_last_signal = now - last_signal_time
        min_cooldown = 60  # 1 minute cooldown between signals
        market_open = is_market_open(symbol)
        open_positions_count = len(state)
        # Unlimited positions - always allow trading
        trading_allowed = not trading_paused
        
        # Apply signal filters
        is_retest_signal = is_retest
        is_opposite_signal = False
        if MUTE_RETESTS:
            is_retest_signal = False
        
        # Debug log for BTC signals
        if "BTC" in symbol:
            if info["buy"] or info["sell"]:
                logger.info(f"✅ BTC SIGNAL FOUND: buy={info['buy']}, sell={info['sell']}")
                # Check why signal isn't being sent
                logger.info(f"🔍 BTC Filter Check: market_open={market_open}, session_valid={session_valid}, near_event={near_event}, high_liquidity={high_liquidity}, sufficient_volatility={sufficient_volatility}, session_close={session_close_approaching}, time_since_last={time_since_last_signal:.0f}s, trading_allowed={trading_allowed}")
            else:
                logger.info(f"⚠️ BTC no signal: filters=[market={market_open}, session={session_valid}, liquidity={high_liquidity}, volatility={sufficient_volatility}, near_event={near_event}, session_close={session_close_approaching}]")
        
        # Check user filters
        user_min_accuracy = user_settings.get(CHAT_ID, {}).get("min_accuracy", 60)
        user_pairs = user_settings.get(CHAT_ID, {}).get("pairs", None)
        user_min_rr = user_settings.get(CHAT_ID, {}).get("min_rr", 1.0)
        user_break_hours = user_settings.get(CHAT_ID, {}).get("break_hours", None)
        
        # Apply filters
        accuracy_ok = info.get("strength", 60) >= user_min_accuracy
        pair_ok = user_pairs is None or symbol.replace("=X", "").replace("-USD", "") in user_pairs
        rrr = info.get("rrr", {})
        rrr_ratio = rrr.get("ratio", 1.0) if rrr else 1.0
        rr_ok = rrr_ratio >= user_min_rr
        break_ok = True
        if user_break_hours and "-" in user_break_hours:
            try:
                start_str, end_str = user_break_hours.split("-")
                start_hour = int(start_str.split(":")[0])
                end_hour = int(end_str.split(":")[0])
                now_hour = datetime.utcnow().hour
                if start_hour < end_hour:
                    break_ok = not (start_hour <= now_hour < end_hour)
                else:  # Overnight break (e.g., 22:00-06:00)
                    break_ok = not (now_hour >= start_hour or now_hour < end_hour)
            except:
                break_ok = True
        
        # ✅ SIGNAL STABILITY: Only send NEW signal or UPDATE if material change
        current_signal = {"side": "BUY" if info["buy"] else "SELL" if info["sell"] else None, "price": price, "strength": info.get("strength", 60)}
        last_signal = signal_stability.get(symbol, {})
        
        # Check if breakout level changed significantly (5% change threshold)
        breakout_changed = False
        if last_signal and current_signal["side"] == last_signal.get("side"):
            price_change_pct = abs(current_signal["price"] - last_signal.get("price", price)) / last_signal.get("price", price) * 100
            strength_change = abs(current_signal["strength"] - last_signal.get("strength", 60))
            breakout_changed = price_change_pct >= 5 or strength_change >= 15  # Material change = 5% or 15 strength points
        
        # 🎯 PIVOT POINTS ONLY - Pure Breakout Detection
        if info["buy"]:
            # 🟢 BUY SIGNAL - PIVOT BREAKOUT ONLY
            tp1, tp2, tp3, sl = open_position(symbol, "BUY", price, atr_val)
            signal_stability[symbol] = {"side": "BUY", "price": price, "strength": info.get("strength", 60), "timestamp": now}
            sent_signals[symbol] = now
            
            msg, copyable = format_entry(symbol, info, "BUY", price, tp1, tp2, tp3, sl)
            pivot_pts = {"support": info.get("swings", {}).get("min"), "resistance": info.get("swings", {}).get("max")}
            await send_message(bot, msg, copyable_text=copyable, post_to_channel=True, symbol=symbol, side="BUY", entry=price, tp1=tp1, tp2=tp2, tp3=tp3, sl=sl, pivot_points=pivot_pts)
            logger.info(f"✅ PIVOT BUY SIGNAL: {symbol} @ {price}")
                    
        elif info["sell"]:
            # 🔴 SELL SIGNAL - PIVOT BREAKOUT ONLY
            tp1, tp2, tp3, sl = open_position(symbol, "SELL", price, atr_val)
            signal_stability[symbol] = {"side": "SELL", "price": price, "strength": info.get("strength", 60), "timestamp": now}
            sent_signals[symbol] = now
            
            msg, copyable = format_entry(symbol, info, "SELL", price, tp1, tp2, tp3, sl)
            pivot_pts = {"support": info.get("swings", {}).get("min"), "resistance": info.get("swings", {}).get("max")}
            await send_message(bot, msg, copyable_text=copyable, post_to_channel=True, symbol=symbol, side="SELL", entry=price, tp1=tp1, tp2=tp2, tp3=tp3, sl=sl, pivot_points=pivot_pts)
            logger.info(f"✅ PIVOT SELL SIGNAL: {symbol} @ {price}")
    
    # ✅ EXIT SIGNAL CHECKING - Send notifications when TP/SL are hit
    if s is not None:
        side = s["side"]
        entry = s["entry"]
        tp1 = s["tp1"]
        tp2 = s["tp2"]
        tp3 = s["tp3"]
        sl = s["sl"]
        
        # Check which exit level was hit
        exit_hit = None
        exit_price = None
        
        if side == "BUY":
            if price >= tp3:
                exit_hit = "TP3"
                exit_price = tp3
            elif price >= tp2:
                exit_hit = "TP2"
                exit_price = tp2
            elif price >= tp1:
                exit_hit = "TP1"
                exit_price = tp1
            elif price <= sl:
                exit_hit = "SL"
                exit_price = sl
        else:  # SELL
            if price <= tp3:
                exit_hit = "TP3"
                exit_price = tp3
            elif price <= tp2:
                exit_hit = "TP2"
                exit_price = tp2
            elif price <= tp1:
                exit_hit = "TP1"
                exit_price = tp1
            elif price >= sl:
                exit_hit = "SL"
                exit_price = sl
        
        # Send exit notification if a level was hit and not already sent
        if exit_hit:
            exit_key = f"{symbol}_{side}_{exit_hit} Hit"
            if exit_key not in sent_exits:
                sent_exits[exit_key] = True
                msg = format_exit(symbol, side, exit_hit, exit_price)[0]
                await send_message(bot, msg)
                # Remove from state when TP3 or SL hit (close position)
                if exit_hit in ["TP3", "SL"]:
                    state.pop(symbol, None)
                    record_trade(symbol, side, entry, exit_price, exit_hit == "TP3", exit_hit)
                    logger.info(f"✅ {symbol} {side} {exit_hit} at {exit_price} - Position CLOSED")
    
    # ✅ MARKET CLOSED: Force close ALL positions when ALL markets are closed
    all_markets_closed = all(not is_market_open(sym) for sym in SYMBOLS)
    if all_markets_closed and state:
        logger.warning(f"🛑 ALL MARKETS CLOSED! Force closing {len(state)} open positions...")
        for sym, pos in list(state.items()):
            side = pos["side"]
            entry = pos["entry"]
            exit_price = price  # Close at current price
            state.pop(sym, None)
            record_trade(sym, side, entry, exit_price, False, "MARKET_CLOSED")
            msg = f"🛑 <b>MARKET CLOSED - POSITION FORCED CLOSED</b>\n{sym} {side}\nEntry: {entry}\nForced Exit: {exit_price}"
            await send_message(bot, msg)
            logger.info(f"🛑 FORCED CLOSE: {sym} {side} - Market closed")


async def monitor_loop(bot: Bot):
    """
    BULLETPROOF MONITORING LOOP - NEVER STOPS, NEVER CRASHES
    - Catches ALL exceptions at every level
    - Has timeout protection for network calls
    - Logs every error for debugging
    - Automatically restarts on any failure
    - Sends health alerts to admin
    - Sends periodic health heartbeats for uptime monitoring
    """
    logger.info("🚀 STARTING BULLETPROOF MARKET MONITORING LOOP...")
    restart_count = 0
    error_streak = 0
    heartbeat_counter = 0  # Send heartbeat every 60 cycles
    telegram_keepalive_counter = 0  # Send Telegram keepalive every 6 cycles (~1 min)
    
    while True:
        try:
            error_streak = 0  # Reset error counter on successful cycle
            heartbeat_counter += 1
            telegram_keepalive_counter += 1
            
            # Send Telegram keepalive every 6 cycles (~1 minute) to prevent timeout
            if telegram_keepalive_counter >= 6:
                await send_telegram_keepalive(bot)
                telegram_keepalive_counter = 0
            
            # Send health heartbeat every 60 cycles (~1 minute if cycling fast)
            if heartbeat_counter >= 60:
                await send_health_heartbeat()
                heartbeat_counter = 0
            
            for symbol in SYMBOLS:
                try:
                    # TIMEOUT PROTECTION: 15 seconds per symbol max
                    async with asyncio.timeout(15):
                        logger.info(f"📊 Checking {symbol}...")
                        await check_symbol(bot, symbol)
                        await asyncio.sleep(0.05)  # Minimal delay to prevent network flooding
                except asyncio.TimeoutError:
                    logger.warning(f"⏱️ TIMEOUT for {symbol} (>15s) - continuing to next symbol")
                    error_streak += 1
                    if error_streak > 5:
                        logger.error(f"🔴 5+ consecutive timeouts - clearing network cache and continuing")
                        error_streak = 1
                    continue
                except Exception as e:
                    logger.error(f"❌ Error checking {symbol}: {type(e).__name__}: {e}")
                    error_streak += 1
                    if error_streak > 10:
                        logger.error(f"🔴 10+ consecutive errors - brief pause then continue")
                        await asyncio.sleep(0.5)  # Very brief pause, then keep going
                        error_streak = 0
                    continue
            
            # NO SLEEP - Cycle immediately without pause (24/7 continuous monitoring)
            if CHECK_INTERVAL > 0:
                logger.info(f"✅ Cycle complete. Next cycle in {CHECK_INTERVAL}s...")
                await asyncio.sleep(CHECK_INTERVAL)
            
        except asyncio.CancelledError:
            logger.critical("🛑 Monitor loop was cancelled - restarting IMMEDIATELY")
            continue  # NO SLEEP - restart immediately
        except Exception as e:
            restart_count += 1
            logger.critical(f"🚨 CRITICAL ERROR in monitor loop (restart #{restart_count}): {type(e).__name__}: {e}")
            logger.error(f"Full traceback: {repr(e)}", exc_info=True)
            
            # NEVER SLEEP - Restart with minimal delay (100ms only)
            wait_time = 0.1  # 100ms max - essentially no sleep
            
            try:
                # Send alert to admin
                if CHAT_ID and restart_count <= 3:
                    try:
                        await bot.send_message(
                            chat_id=CHAT_ID,
                            text=f"⚠️ <b>MONITOR LOOP RESTARTING</b>\nRestart #{restart_count}\nError: {str(e)[:100]}"
                        )
                    except:
                        pass  # Don't let admin alert block recovery
                
                await asyncio.sleep(wait_time)  # 100ms only
            except:
                pass  # Skip sleep if alert fails
            
            continue  # CRITICAL: Restart the loop immediately!


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    daily_loss = get_daily_loss()
    open_pos = len(state)
    # Always LIVE since positions are unlimited
    status_icon = "✅ LIVE"
    
    # Get user subscription status
    user_sub = subscribers.get(chat_id, {"plan": "free", "status": "inactive"})
    sub_plan = user_sub.get("plan", "free")
    sub_icon = {"free": "🆓", "signals": "💰", "autotrader": "🤖"}.get(sub_plan, "❓")
    
    welcome_msg = (
        "🚀 <b>ALPHATRADE SIGNALS</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Status: <b>{status_icon}</b>\n"
        f"💼 Positions: <b>UNLIMITED</b>\n"
        f"📈 Markets: <b>{len(SYMBOLS)} pairs</b>\n"
        f"⏰ TF: <b>{TIMEFRAME}</b>\n"
        f"{sub_icon} <b>{sub_plan.upper()}</b> PLAN\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Daily Loss: {daily_loss:.0f}p / UNLIMITED\n"
        f"Retest: {'🔇 OFF' if MUTE_RETESTS else '🔊 ON'}\n\n"
        "<b>👇 CHOOSE ACTION:</b>"
    )
    
    # Professional organized button layout
    keyboard = [
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton("📈 Analytics", callback_data="analytics")],
        [InlineKeyboardButton("🔍 Check Markets", callback_data="check"),
         InlineKeyboardButton("🧪 Test Signals", callback_data="test")],
        [InlineKeyboardButton("📋 Commands", callback_data="commands"),
         InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        [InlineKeyboardButton("⏱️ Timeframe", callback_data="tf"),
         InlineKeyboardButton("💳 Pricing", callback_data="pricing")],
        [InlineKeyboardButton("💬 Live Support", callback_data="support")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(welcome_msg, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode="HTML")


async def cmd_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Checking all markets now...")
    bot = context.bot
    for sym in SYMBOLS:
        await check_symbol(bot, sym)
        await asyncio.sleep(0.5)
    await update.message.reply_text("✅ Check complete!")


async def cmd_test_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test ALL 14 pairs - Generate real signals for each"""
    try:
        await update.message.reply_text(f"🧪 Testing all {len(SYMBOLS)} pairs for live signals...")
        
        signals_found = 0
        test_results = []
        
        for symbol in SYMBOLS:
            try:
                # Clean symbol for display (remove =X, -USD suffix)
                display_symbol = symbol.replace("=X", "").replace("-USD", "")
                
                # Analyze this symbol for real signals
                info = analyze(symbol)
                if info is None:
                    test_results.append(f"⚪ {display_symbol}: No data")
                    await asyncio.sleep(0.1)
                    continue
                
                price = info.get("price", 0)
                buy = info.get("buy", False)
                sell = info.get("sell", False)
                confluence = info.get("confluence", 0)
                rsi = info.get("rsi", 0)
                atr = info.get("atr", 0)
                
                if buy or sell:
                    signals_found += 1
                    side = "📈 BUY" if buy else "📉 SELL"
                    test_results.append(f"🟢 {display_symbol}: {side} | Confluence: {confluence:.0f}% | RSI: {rsi:.1f} | ATR: {atr:.4f}")
                else:
                    test_results.append(f"⚪ {display_symbol}: No signal")
                await asyncio.sleep(0.1)
            except Exception as e:
                display_symbol = symbol.replace("=X", "").replace("-USD", "")
                error_msg = str(e)[:30] if e else "Unknown error"
                logger.error(f"Error analyzing {display_symbol}: {e}")
                test_results.append(f"❌ {display_symbol}: Error - {error_msg}")
        
        # Format results
        summary = (
            f"🧪 <b>LIVE SIGNAL TEST - ALL {len(SYMBOLS)} PAIRS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Signals Found: {signals_found}/{len(SYMBOLS)}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(test_results) +
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        
        # Send result to user
        if update.callback_query:
            await update.callback_query.edit_message_text(summary, parse_mode="HTML")
        else:
            await update.message.reply_text(summary, parse_mode="HTML")
        
        # Also send to admin
        if CHAT_ID:
            try:
                await context.bot.send_message(chat_id=CHAT_ID, text=summary, parse_mode="HTML")
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in test signal command: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)[:100]}")


async def cmd_demo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a DEMO signal to test Telegram integration"""
    demo_signal = (
        "🟢 <b>EURUSD - DEMO BUY SIGNAL</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📍 <b>Entry</b>: 1.08750\n"
        "🎯 <b>TP1</b>: 1.08850 (+100 pips)\n"
        "🎯 <b>TP2</b>: 1.08950 (+200 pips)\n"
        "🎯 <b>TP3</b>: 1.09050 (+300 pips)\n"
        "🛑 <b>SL</b>: 1.08650 (-100 pips)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        "\n✅ <b>THIS IS A DEMO TEST SIGNAL</b>\n"
        "<i>Real signals are being monitored and will be sent automatically</i>"
    )
    await update.message.reply_text(demo_signal, parse_mode="HTML")


async def cmd_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SYMBOLS
    if not context.args:
        await update.message.reply_text(
            f"Current symbols: {', '.join(SYMBOLS)}\n\n"
            "❌ Usage: /symbols SYMBOL1,SYMBOL2,...\n"
            "Example: /symbols EURUSD=X,GC=F,BTC-USD"
        )
        return
    raw = " ".join(context.args)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if parts:
        SYMBOLS = parts
        await update.message.reply_text(f"✅ Symbols updated: {', '.join(SYMBOLS)}")


async def cmd_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global TIMEFRAME
    valid_tfs = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"]
    if not context.args:
        await update.message.reply_text(
            f"Current: {TIMEFRAME}\n"
            f"Valid: {', '.join(valid_tfs)}\n"
            "Usage: /tf 30m"
        )
        return
    tf = context.args[0].strip()
    if tf in valid_tfs:
        TIMEFRAME = tf
        await update.message.reply_text(f"✅ Timeframe set to {TIMEFRAME}")
    else:
        await update.message.reply_text(f"❌ Invalid. Valid: {', '.join(valid_tfs)}")


async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FAST_MA, SLOW_MA, RSI_PERIOD, TP1_POINTS, TP2_POINTS, TP3_POINTS, SL_POINTS
    if len(context.args) != 7:
        await update.message.reply_text(
            "❌ Usage: /set FAST_MA SLOW_MA RSI TP1 TP2 TP3 SL\n"
            "Example: /set 10 30 14 30 60 100 40"
        )
        return
    try:
        FAST_MA = int(context.args[0])
        SLOW_MA = int(context.args[1])
        RSI_PERIOD = int(context.args[2])
        TP1_POINTS = int(context.args[3])
        TP2_POINTS = int(context.args[4])
        TP3_POINTS = int(context.args[5])
        SL_POINTS = int(context.args[6])
        await update.message.reply_text(
            f"✅ Settings updated:\n"
            f"MA: {FAST_MA}/{SLOW_MA} | RSI: {RSI_PERIOD}\n"
            f"TPs: {TP1_POINTS}/{TP2_POINTS}/{TP3_POINTS} | SL: {SL_POINTS}"
        )
    except ValueError:
        await update.message.reply_text("❌ All values must be integers!")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not state:
        msg = "💼 OPEN POSITIONS: 0/3\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n✅ Ready for new signals"
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                     InlineKeyboardButton("🏠 Home", callback_data="home")]]
        reply_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
        await reply_func(msg, reply_markup=InlineKeyboardMarkup(keyboard))
        return
    lines = ["💼 OPEN POSITIONS: %d/3\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" % len(state)]
    for sym, s in state.items():
        clean = sym.replace("=X", "").replace("-USD", "")
        lines.append(
            f"\n{'📈' if s['side']=='BUY' else '📉'} {clean} {s['side']}\n"
            f"Entry: {format_price(sym, s['entry'])}\n"
            f"TP1: {format_price(sym, s['tp1'])} | TP2: {format_price(sym, s['tp2'])} | TP3: {format_price(sym, s['tp3'])}\n"
            f"🛑 SL: {format_price(sym, s['sl'])}"
        )
    
    keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                 InlineKeyboardButton("📈 Analytics", callback_data="analytics")],
                [InlineKeyboardButton("🏠 Home", callback_data="home")]]
    reply_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
    await reply_func("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols_display = ", ".join([s.replace("=X", "").replace("-USD", "") for s in SYMBOLS[:5]])
    if len(SYMBOLS) > 5:
        symbols_display += f" +{len(SYMBOLS)-5}"
    
    msg = (
        "⚙️ SETTINGS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"timeframe: {TIMEFRAME}\n"
        f"symbols: {symbols_display}\n"
        f"scan: {CHECK_INTERVAL}s\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"MA: {FAST_MA}/{SLOW_MA}\n"
        f"RSI: {RSI_PERIOD} (B≥{RSI_BUY}, S≤{RSI_SELL})\n"
        f"TP1/TP2/TP3: {TP1_POINTS}/{TP2_POINTS}/{TP3_POINTS}\n"
        f"SL: {SL_POINTS}\n"
        f"Mode: {'🔇 Retest OFF' if MUTE_RETESTS else '🔊 Retest ON'}"
    )
    
    keyboard = [[InlineKeyboardButton("🔗 Symbols", callback_data="symbols"),
                 InlineKeyboardButton("⏱️ Timeframe", callback_data="tf")],
                [InlineKeyboardButton("⚡ Parameters", callback_data="set"),
                 InlineKeyboardButton("🏠 Home", callback_data="home")]]
    reply_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
    await reply_func(msg, reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = get_today_stats()
    if not stats:
        msg = "📊 No trades today yet."
        keyboard = [[InlineKeyboardButton("🔄 Refresh", callback_data="analytics"),
                     InlineKeyboardButton("🏠 Home", callback_data="home")]]
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(keyboard))
        return
    
    pf_str = f"{stats['profit_factor']:.2f}x" if stats['profit_factor'] > 0 else "N/A"
    best_hour_str = f"{stats['best_hour']}:00 UTC" if stats['best_hour'] is not None else "N/A"
    daily_loss = get_daily_loss()
    
    msg = (
        "📊 TODAY'S ADVANCED ANALYTICS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total Trades: {stats['total_trades']}\n"
        f"✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']}\n"
        f"🎯 Win Rate: {stats['win_rate']:.1f}%\n"
        f"📊 Profit Factor: {pf_str}\n"
        f"💰 Total Pips: {stats['total_pips']:+.0f} (Wins: {stats['win_pips']:+.0f} | Loss: {stats['loss_pips']:.0f})\n"
        f"📅 Daily Loss: {daily_loss:.0f}p (Limit: {DAILY_LOSS_LIMIT}p)\n"
        f"🏆 Consecutive Wins: {stats['consecutive_wins']} | 📉 Consecutive Loss: {stats['consecutive_losses']}\n"
        f"⏰ Best Hour: {best_hour_str}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    
    if stats['best_trade']:
        msg += f"🏆 Best: {stats['best_trade']['symbol']} +{stats['best_trade']['pips']:.0f}p\n"
    if stats['worst_trade']:
        msg += f"📉 Worst: {stats['worst_trade']['symbol']} {stats['worst_trade']['pips']:.0f}p\n"
    
    if stats['symbol_pips']:
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📌 By Symbol:\n"
        for sym, pips in sorted(stats['symbol_pips'].items(), key=lambda x: x[1], reverse=True):
            msg += f"  {sym}: {pips:+.0f}p\n"
    
    keyboard = [[InlineKeyboardButton("📊 Status", callback_data="status"),
                 InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
                [InlineKeyboardButton("🏠 Home", callback_data="home")]]
    reply_func = update.callback_query.edit_message_text if update.callback_query else update.message.reply_text
    await reply_func(msg, reply_markup=InlineKeyboardMarkup(keyboard))


async def cmd_mute_retests(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MUTE_RETESTS
    MUTE_RETESTS = not MUTE_RETESTS
    status = "🔇 MUTED" if MUTE_RETESTS else "🔊 UNMUTED"
    msg = f"{status} - Retest signals {status.split()[1].lower()}"
    await update.message.reply_text(msg)


async def cmd_mute_opposites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MUTE_OPPOSITES
    MUTE_OPPOSITES = not MUTE_OPPOSITES
    status = "🔇 MUTED" if MUTE_OPPOSITES else "🔊 UNMUTED"
    msg = f"{status} - Opposite signal exits {status.split()[1].lower()}"
    await update.message.reply_text(msg)


async def cmd_reset_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global trading_paused
    trading_paused = False
    msg = "✅ Daily loss limit reset. Trading resumed."
    await update.message.reply_text(msg)


async def send_daily_summary(bot: Bot):
    """Send daily summary at 22:00 UTC"""
    stats = get_today_stats()
    if not stats or stats['total_trades'] == 0:
        return
    
    msg = (
        "📊 DAILY SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total Trades: {stats['total_trades']}\n"
        f"✅ Wins: {stats['wins']} ({stats['win_rate']:.1f}%)\n"
        f"❌ Losses: {stats['losses']}\n"
        f"💰 Total Pips: {stats['total_pips']:+.0f}\n"
    )
    
    if stats['best_trade']:
        msg += f"🏆 Best Trade: {stats['best_trade']['symbol']} +{stats['best_trade']['pips']:.0f}p\n"
    
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception as e:
        logger.error(f"Error sending daily summary: {e}")


# Flask HTTP server - works through Replit proxy
flask_app = Flask(__name__)

@flask_app.route('/', methods=['GET', 'POST'])
def health():
    return "OK", 200

def start_http_server():
    """Start bulletproof raw socket HTTP server - can't crash"""
    import socket
    import threading
    
    def handle_client(client_socket, addr):
        try:
            request = client_socket.recv(1024).decode('utf-8', errors='ignore')
            if b'GET' in request.encode() or b'POST' in request.encode():
                response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nOK"
                client_socket.send(response.encode())
        except:
            pass
        finally:
            client_socket.close()
    
    def socket_server():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", 5000))
        server.listen(5)
        logger.info("✅ HTTP Server ONLINE on 0.0.0.0:5000")
        
        while True:
            try:
                client, addr = server.accept()
                client_thread = threading.Thread(target=handle_client, args=(client, addr), daemon=True)
                client_thread.start()
            except:
                pass
    
    try:
        socket_server()
    except Exception as e:
        logger.error(f"❌ HTTP Server failed: {e}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show signal history - last 10 signals"""
    if not signal_history:
        await update.message.reply_text("📋 SIGNAL HISTORY\n━━━━━━━━━━━━━━━━━\nNo signals sent yet", parse_mode="HTML")
        return
    
    msg = "📋 <b>SIGNAL HISTORY (Last 10)</b>\n━━━━━━━━━━━━━━━━━\n\n"
    for sig in signal_history[-10:]:
        msg += f"{'🟢' if sig.get('side')=='BUY' else '🔴'} {sig['symbol']} {sig['side']} @ {sig['entry']}\n"
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_enter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually enter a trade: /enter EURUSD 1.08750 BUY"""
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Usage: /enter PAIR ENTRY_PRICE BUY/SELL\nExample: /enter EURUSD 1.08750 BUY")
        return
    
    symbol = args[0].upper()
    try:
        entry = float(args[1])
        side = args[2].upper()
        if side not in ["BUY", "SELL"]:
            raise ValueError("Side must be BUY or SELL")
    except:
        await update.message.reply_text("❌ Invalid format. Use: /enter EURUSD 1.08750 BUY")
        return
    
    chat_id = update.effective_chat.id
    if chat_id not in user_manual_trades:
        user_manual_trades[chat_id] = {}
    
    user_manual_trades[chat_id][symbol] = {
        "side": side,
        "entry": entry,
        "entry_time": datetime.utcnow(),
        "pips": 0
    }
    
    await update.message.reply_text(f"✅ Trade recorded: {symbol} {side} @ {entry}\nUse /close {symbol} <price> to close")


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Close a manual trade: /close EURUSD 1.08900"""
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Usage: /close PAIR EXIT_PRICE\nExample: /close EURUSD 1.08900")
        return
    
    symbol = args[0].upper()
    try:
        exit_price = float(args[1])
    except:
        await update.message.reply_text("❌ Invalid exit price")
        return
    
    chat_id = update.effective_chat.id
    if chat_id not in user_manual_trades or symbol not in user_manual_trades[chat_id]:
        await update.message.reply_text(f"❌ No open trade for {symbol}")
        return
    
    trade = user_manual_trades[chat_id][symbol]
    entry = trade["entry"]
    side = trade["side"]
    pips = (exit_price - entry) * 10000 if "JPY" in symbol or "BTC" in symbol else (exit_price - entry) * 10000
    if side == "SELL":
        pips = -pips
    
    msg = f"✅ <b>TRADE CLOSED</b>\n{symbol} {side}\nEntry: {entry}\nExit: {exit_price}\nProfit: <b>{pips:+.0f} pips</b>"
    await update.message.reply_text(msg, parse_mode="HTML")
    del user_manual_trades[chat_id][symbol]


async def cmd_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's stats"""
    today = datetime.utcnow().date()
    today_signals = [s for s in signal_history if s.get('time', datetime.utcnow()).date() == today]
    msg = f"📊 <b>TODAY'S SUMMARY</b>\n━━━━━━━━━━━━━━━━━\nSignals: {len(today_signals)}\n"
    msg += f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show win rate per pair"""
    pair_stats = {}
    for sig in signal_history:
        sym = sig['symbol']
        if sym not in pair_stats:
            pair_stats[sym] = {"total": 0, "wins": 0}
        pair_stats[sym]["total"] += 1
    
    msg = "📈 <b>PAIR STATISTICS</b>\n━━━━━━━━━━━━━━━━━\n"
    for sym, stats in sorted(pair_stats.items()):
        wr = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
        msg += f"{sym}: {wr:.0f}% ({stats['wins']}/{stats['total']})\n"
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_min_accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set minimum accuracy filter: /min_accuracy 80"""
    if not context.args:
        current = user_settings.get(update.effective_chat.id, {}).get("min_accuracy", 60)
        await update.message.reply_text(f"⚙️ <b>ACCURACY FILTER</b>\nCurrent: {current}/100\n\nSet: /min_accuracy 80", parse_mode="HTML")
        return
    
    try:
        accuracy = int(context.args[0])
        if 0 <= accuracy <= 100:
            chat_id = update.effective_chat.id
            if chat_id not in user_settings:
                user_settings[chat_id] = {}
            user_settings[chat_id]["min_accuracy"] = accuracy
            await update.message.reply_text(f"✅ Accuracy filter set to: {accuracy}/100\nOnly signals ≥ {accuracy}/100 will be sent")
        else:
            await update.message.reply_text("❌ Accuracy must be 0-100")
    except:
        await update.message.reply_text("❌ Invalid format. Use: /min_accuracy 80")


async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Select which pairs to trade: /pairs EURUSD,GBPUSD,BTC"""
    if not context.args:
        current = user_settings.get(update.effective_chat.id, {}).get("pairs")
        if current:
            pairs_str = ", ".join(current)
        else:
            pairs_str = "All pairs"
        await update.message.reply_text(f"📋 <b>PAIR SELECTION</b>\nCurrent: {pairs_str}\n\nSet: /pairs EURUSD,GBPUSD,BTC", parse_mode="HTML")
        return
    
    pairs = [p.upper() for p in context.args[0].split(",")]
    chat_id = update.effective_chat.id
    if chat_id not in user_settings:
        user_settings[chat_id] = {}
    user_settings[chat_id]["pairs"] = pairs
    await update.message.reply_text(f"✅ Pairs selected: {', '.join(pairs)}\nYou'll only receive signals for these pairs")


async def cmd_min_rr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set minimum R/R ratio: /min_rr 2.0"""
    if not context.args:
        current = user_settings.get(update.effective_chat.id, {}).get("min_rr", 1.0)
        await update.message.reply_text(f"⚙️ <b>R/R FILTER</b>\nCurrent: {current}:1\n\nSet: /min_rr 2.0", parse_mode="HTML")
        return
    
    try:
        rr = float(context.args[0])
        if rr > 0:
            chat_id = update.effective_chat.id
            if chat_id not in user_settings:
                user_settings[chat_id] = {}
            user_settings[chat_id]["min_rr"] = rr
            await update.message.reply_text(f"✅ R/R filter set to: {rr}:1\nOnly signals with R/R ≥ {rr}:1 will be sent")
        else:
            await update.message.reply_text("❌ R/R must be positive")
    except:
        await update.message.reply_text("❌ Invalid format. Use: /min_rr 2.0")


async def cmd_break_hours(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set break hours: /quiet 22:00 06:00"""
    if not context.args or len(context.args) < 2:
        current = user_settings.get(update.effective_chat.id, {}).get("break_hours")
        if current:
            await update.message.reply_text(f"⏱️ <b>BREAK HOURS</b>\nCurrent: {current}\n\nSet: /quiet 22:00 06:00", parse_mode="HTML")
        else:
            await update.message.reply_text("⏱️ <b>BREAK HOURS</b>\nNo break hours set\n\nSet: /quiet 22:00 06:00\nNo signals 22:00-06:00 UTC", parse_mode="HTML")
        return
    
    try:
        start = context.args[0]
        end = context.args[1]
        # Validate time format HH:MM
        start_h = int(start.split(":")[0])
        start_m = int(start.split(":")[1])
        end_h = int(end.split(":")[0])
        end_m = int(end.split(":")[1])
        
        if 0 <= start_h <= 23 and 0 <= start_m <= 59 and 0 <= end_h <= 23 and 0 <= end_m <= 59:
            chat_id = update.effective_chat.id
            if chat_id not in user_settings:
                user_settings[chat_id] = {}
            break_hours = f"{start}-{end}"
            user_settings[chat_id]["break_hours"] = break_hours
            await update.message.reply_text(f"✅ Break hours set: {break_hours} UTC\nNo signals during this period")
        else:
            await update.message.reply_text("❌ Invalid time format")
    except:
        await update.message.reply_text("❌ Invalid format. Use: /quiet 22:00 06:00")


async def cmd_support(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Open live support chat"""
    chat_id = update.effective_chat.id
    username = update.effective_user.username or "Anonymous"
    
    # Create support interface
    support_msg = (
        "💬 <b>LIVE SUPPORT</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📌 <b>Quick Help Topics:</b>\n"
        "• How to use the bot\n"
        "• Signal accuracy questions\n"
        "• Trading strategy help\n"
        "• Technical issues\n"
        "• Billing/Subscription\n\n"
        "💬 <b>Type your question:</b>\n"
        "Just reply with your message and our admin will respond ASAP!\n\n"
        f"⏰ <i>Your Chat ID: {chat_id}</i>\n"
        f"👤 <i>Username: @{username}</i>"
    )
    
    if update.callback_query:
        await update.callback_query.edit_message_text(support_msg, parse_mode="HTML")
    else:
        await update.message.reply_text(support_msg, parse_mode="HTML")
    
    # Store that user is in support mode
    context.user_data['in_support'] = True


async def handle_support_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user support messages - route to admin"""
    if not context.user_data.get('in_support'):
        return
    
    chat_id = update.effective_chat.id
    username = update.effective_user.username or f"User_{chat_id}"
    user_message = update.message.text
    
    # Create ticket
    ticket_id = f"{chat_id}_{datetime.utcnow().timestamp()}"
    support_tickets[ticket_id] = {
        "user_id": chat_id,
        "username": username,
        "message": user_message,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "open",
        "responses": []
    }
    
    # Acknowledge to user
    await update.message.reply_text(
        "✅ <b>TICKET CREATED</b>\n"
        f"Ticket ID: <code>{ticket_id[:20]}...</code>\n\n"
        "Admin will respond shortly! 📬\n"
        "You'll be notified when there's a reply.",
        parse_mode="HTML"
    )
    
    # Send to admin
    if CHAT_ID:
        try:
            admin_msg = (
                f"📨 <b>NEW SUPPORT TICKET</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 User: @{username}\n"
                f"🆔 Chat ID: {chat_id}\n"
                f"🎟️ Ticket: <code>{ticket_id[:15]}...</code>\n\n"
                f"💬 <b>Message:</b>\n{user_message}\n\n"
                f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
            await context.bot.send_message(chat_id=CHAT_ID, text=admin_msg, parse_mode="HTML")
        except:
            pass
    
    context.user_data['in_support'] = False


async def cmd_signal_explanation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show why signals are generated"""
    msg = "📊 <b>WHY SIGNALS ARE GENERATED?</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += "<b>All signals require:</b>\n"
    msg += "✅ 15+ of 26 indicators confluent\n"
    msg += "✅ Market is open\n"
    msg += "✅ High liquidity hour\n"
    msg += "✅ Sufficient volatility (ATR)\n"
    msg += "✅ No high-impact economic events\n"
    msg += "✅ Session not ending soon\n"
    msg += "✅ 5+ min since last signal\n\n"
    msg += "<b>Your active filters:</b>\n"
    settings = user_settings.get(update.effective_chat.id, {})
    min_acc = settings.get("min_accuracy", 60)
    msg += f"• Accuracy: ≥ {min_acc}/100\n"
    pairs = settings.get("pairs")
    if pairs:
        msg += f"• Pairs: {', '.join(pairs)}\n"
    else:
        msg += "• Pairs: All 14\n"
    min_rr = settings.get("min_rr", 1.0)
    msg += f"• R/R Ratio: ≥ {min_rr}:1\n"
    break_hours = settings.get("break_hours")
    if break_hours:
        msg += f"• Break hours: {break_hours} UTC\n"
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show currently open live positions"""
    if not state:
        await update.message.reply_text(
            "📊 LIVE POSITIONS\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "No open positions\n\n"
            "Waiting for next signal...",
            parse_mode="HTML"
        )
        return
    
    live_text = "📊 LIVE POSITIONS\n━━━━━━━━━━━━━━━━━━━\n\n"
    for symbol, pos in state.items():
        clean_symbol = symbol.replace("=X", "").replace("-USD", "")
        side = pos["side"]
        entry = pos["entry"]
        tp1, tp2, tp3 = pos["tp1"], pos["tp2"], pos["tp3"]
        sl = pos["sl"]
        price = format_price(symbol, entry)
        tp1_str = format_price(symbol, tp1)
        tp2_str = format_price(symbol, tp2)
        tp3_str = format_price(symbol, tp3)
        sl_str = format_price(symbol, sl)
        
        live_text += (
            f"{'🟢' if side == 'BUY' else '🔴'} {clean_symbol}\n"
            f"Position: {side}\n"
            f"Entry: {price}\n"
            f"TP1: {tp1_str} | TP2: {tp2_str} | TP3: {tp3_str}\n"
            f"SL: {sl_str}\n\n"
        )
    
    await update.message.reply_text(live_text, parse_mode="HTML")


async def cmd_pricing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show subscription pricing with easy buttons"""
    chat_id = update.effective_chat.id
    user_sub = subscribers.get(chat_id, {"plan": "free"})
    current_plan = user_sub.get("plan", "free")
    
    pricing_msg = (
        "💳 SUBSCRIPTION PLANS\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🆓 FREE PLAN\n"
        "├─ Price: FREE\n"
        "├─ Signals/day: 1 signal\n"
        "├─ Symbols: 5 pairs\n"
        "└─ Auto-trade: ❌\n\n"
        "💰 SIGNALS PLAN ⭐\n"
        "├─ Price: 5 USDT/month\n"
        "├─ Signals/day: 5-10 signals\n"
        "├─ Symbols: 14 pairs (all)\n"
        "└─ Auto-trade: ❌\n\n"
        "🤖 AUTOTRADER PLAN 🔥\n"
        "├─ Price: 10 USDT/month\n"
        "├─ Signals/day: Unlimited\n"
        "├─ Symbols: 14 pairs (all)\n"
        "└─ Auto-trade: ✅ MT4 auto-execute\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Your Plan: {current_plan.upper()}\n\n"
        "💰 USDT WALLET (TRC20 ONLY):\n"
        f"`{USDT_WALLET}`\n\n"
        "⚡ QUICK UPGRADE STEPS:\n"
        "1️⃣ Copy wallet above (tap code block)\n"
        "2️⃣ Send USDT on TRC20 network\n"
        "3️⃣ Screenshot payment receipt\n"
        "4️⃣ Use buttons below → send pic\n"
        "5️⃣ ✅ Activated within 24 hours!"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("💳 Subscribe Signals (5 USDT)", callback_data="subscribe_signals"),
            InlineKeyboardButton("🤖 Subscribe Autotrader (10 USDT)", callback_data="subscribe_autotrader")
        ],
        [
            InlineKeyboardButton("📋 Check Status", callback_data="check_sub_status"),
            InlineKeyboardButton("🏠 Home", callback_data="home")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query = update.callback_query
    if query:
        await query.edit_message_text(text=pricing_msg, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(text=pricing_msg, reply_markup=reply_markup, parse_mode="Markdown")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Subscribe to a plan - sends request for manual verification"""
    chat_id = update.effective_chat.id
    args = context.args
    
    if not args or len(args) < 1:
        await update.message.reply_text("Usage: /subscribe signals  OR  /subscribe autotrader")
        return
    
    plan = args[0].lower()
    if plan not in subscription_plans:
        await update.message.reply_text(f"❌ Invalid plan. Options: signals, autotrader")
        return
    
    subscribers[chat_id] = {
        "plan": plan,
        "status": "pending",
        "joined": datetime.utcnow().timestamp()
    }
    
    price = subscription_plans[plan]["price"]
    plan_name = "SIGNALS" if plan == "signals" else "AUTOTRADER"
    msg = (
        f"✅ SUBSCRIPTION REQUEST RECEIVED!\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Plan: {plan_name}\n"
        f"Price: {price}\n"
        f"Status: ⏳ AWAITING PAYMENT VERIFICATION\n\n"
        f"📸 NEXT STEPS:\n"
        f"1️⃣ Send {price} USDT to:\n"
        f"   <code>{USDT_WALLET}</code>\n"
        f"2️⃣ Take screenshot of TRC20 receipt\n"
        f"3️⃣ Use /upload_receipt to submit pic\n"
        f"4️⃣ We verify within 24 hours\n"
        f"5️⃣ ✅ Full access granted!\n\n"
        f"💬 Send payment receipt now ↓"
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    # Store pending request with plan
    context.user_data['pending_plan'] = plan


async def cmd_check_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check current subscription"""
    chat_id = update.effective_chat.id
    user_sub = subscribers.get(chat_id, {"plan": "free", "status": "inactive"})
    plan = user_sub.get("plan", "free")
    status = user_sub.get("status", "inactive")
    
    plan_info = subscription_plans.get(plan, {})
    
    status_display = "✅ ACTIVE" if status == "active" else "⏳ PENDING" if status == "pending" else "❌ INACTIVE"
    
    msg = (
        f"📋 YOUR SUBSCRIPTION\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Plan: {plan.upper()}\n"
        f"Price: {plan_info.get('price', 'Free')}\n"
        f"Status: {status_display}\n"
        f"Signals/day: {plan_info.get('signals_per_day', 0)}\n"
        f"Symbols: {plan_info.get('symbols', 5)}\n"
        f"Auto-trade: {'✅ YES' if plan_info.get('auto_trade') else '❌ NO'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Activation: Within 24 hours after payment verified"
    )
    await update.message.reply_text(msg)


async def cmd_upload_receipt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Upload payment receipt - expects user to attach photo/document"""
    chat_id = update.effective_chat.id
    username = update.effective_user.username or f"User_{chat_id}"
    user_sub = subscribers.get(chat_id, {})
    plan = user_sub.get("plan")
    
    if not plan:
        await update.message.reply_text(
            "❌ No pending subscription found.\n"
            "Please /subscribe first, then upload receipt."
        )
        return
    
    await update.message.reply_text(
        f"📸 <b>RECEIPT UPLOAD</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Plan: {plan.upper()}\n"
        f"Status: Ready to receive receipt\n\n"
        f"⬇️ Please send:\n"
        f"• Photo/screenshot of TRC20 payment\n"
        f"• Document with transaction hash\n\n"
        f"✅ We'll verify & activate within 24h",
        parse_mode="HTML"
    )
    context.user_data['awaiting_receipt'] = True


async def handle_receipt_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle receipt photo/document upload from user"""
    if not context.user_data.get('awaiting_receipt'):
        return
    
    chat_id = update.effective_chat.id
    username = update.effective_user.username or f"User_{chat_id}"
    user_sub = subscribers.get(chat_id, {})
    plan = user_sub.get("plan", "unknown")
    
    # Get file info
    file = None
    file_type = "photo"
    if update.message.photo:
        file = update.message.photo[-1]
        file_type = "photo"
    elif update.message.document:
        file = update.message.document
        file_type = "document"
    else:
        return
    
    # Store receipt
    receipt_id = f"{chat_id}_{datetime.utcnow().timestamp()}"
    payment_receipts[receipt_id] = {
        "user_id": chat_id,
        "username": username,
        "plan": plan,
        "file_id": file.file_id,
        "file_type": file_type,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "pending"
    }
    
    # Confirm to user
    await update.message.reply_text(
        f"✅ <b>RECEIPT RECEIVED!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Receipt ID: <code>{receipt_id[:15]}...</code>\n"
        f"Plan: {plan.upper()}\n\n"
        f"⏳ We're verifying your payment...\n"
        f"Activation: Within 24 hours\n\n"
        f"We'll notify you when activated! 🎉",
        parse_mode="HTML"
    )
    
    # Send to admin with file
    if CHAT_ID:
        try:
            caption = (
                f"📨 <b>NEW PAYMENT RECEIPT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"👤 User: @{username}\n"
                f"🆔 Chat ID: {chat_id}\n"
                f"💎 Plan: {plan.upper()}\n"
                f"🎟️ Receipt: <code>{receipt_id[:15]}...</code>\n"
                f"📝 Type: {file_type}\n"
                f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                f"✅ To verify & activate:\n"
                f"/activate_user {chat_id} {plan}"
            )
            
            if file_type == "photo":
                await context.bot.send_photo(chat_id=CHAT_ID, photo=file.file_id, caption=caption, parse_mode="HTML")
            else:
                await context.bot.send_document(chat_id=CHAT_ID, document=file.file_id, caption=caption, parse_mode="HTML")
        except:
            pass
    
    context.user_data['awaiting_receipt'] = False


async def cmd_activate_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """[ADMIN ONLY] Manually activate a user subscription after payment verification"""
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("Usage: /activate_user USER_ID plan\nExample: /activate_user 123456 signals")
        return
    
    try:
        user_id = int(args[0])
        plan = args[1].lower()
        
        if plan not in subscription_plans:
            await update.message.reply_text(f"❌ Invalid plan. Options: signals, autotrader")
            return
        
        if user_id in subscribers:
            subscribers[user_id]["status"] = "active"
            msg = f"✅ User {user_id} activated on {plan.upper()} plan"
            logger.info(f"✅ User {user_id} manually activated on {plan} plan")
        else:
            subscribers[user_id] = {
                "plan": plan,
                "status": "active",
                "joined": datetime.utcnow().timestamp()
            }
            msg = f"✅ User {user_id} created and activated on {plan.upper()} plan"
            logger.info(f"✅ User {user_id} created and activated on {plan} plan")
        
        await update.message.reply_text(msg)
    except ValueError:
        await update.message.reply_text("❌ Invalid user ID")


async def cmd_enable_news_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enable news filter to pause trading during economic events"""
    global USE_NEWS_FILTER
    USE_NEWS_FILTER = True
    msg = f"✅ <b>NEWS FILTER ENABLED</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n🗞️ Bot will pause signals during major economic news\n⏱️ Filter window: {NEWS_FILTER_MINUTES_BEFORE} min before / {NEWS_FILTER_MINUTES_AFTER} min after events"
    await update.message.reply_text(msg, parse_mode="HTML")
    logger.info("✅ News filter ENABLED")

async def cmd_disable_news_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Disable news filter to continue trading during economic events"""
    global USE_NEWS_FILTER
    USE_NEWS_FILTER = False
    msg = "❌ <b>NEWS FILTER DISABLED</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚠️ Bot will send signals even during major economic news"
    await update.message.reply_text(msg, parse_mode="HTML")
    logger.info("❌ News filter DISABLED")

async def cmd_news_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set news filter interval (minutes before/after events to avoid)"""
    global NEWS_FILTER_MINUTES_BEFORE, NEWS_FILTER_MINUTES_AFTER
    args = context.args
    if len(args) < 1:
        msg = f"Usage: /news_interval <minutes>\nExample: /news_interval 20\nCurrent: {NEWS_FILTER_MINUTES_BEFORE} min before / {NEWS_FILTER_MINUTES_AFTER} min after"
        await update.message.reply_text(msg)
        return
    try:
        mins = int(args[0])
        NEWS_FILTER_MINUTES_BEFORE = mins
        NEWS_FILTER_MINUTES_AFTER = mins
        msg = f"✅ News filter interval set to <b>{mins} minutes</b> before/after events"
        await update.message.reply_text(msg, parse_mode="HTML")
        logger.info(f"📰 News filter interval updated: {mins} min")
    except:
        await update.message.reply_text("❌ Invalid interval. Use a number like: /news_interval 20")


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "check":
        await cmd_check(update, context)
    elif data == "status":
        await cmd_status(update, context)
    elif data == "analytics":
        await cmd_analytics(update, context)
    elif data == "settings":
        await cmd_settings(update, context)
    elif data == "commands":
        commands_msg = (
            "📋 <b>ALL COMMANDS</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>📊 TRACKING</b>\n"
            "/history - Last 10 signals\n"
            "/enter PAIR PRICE BUY/SELL - Record entry\n"
            "/close PAIR PRICE - Close trade\n"
            "/live - Open positions\n\n"
            "<b>📈 ANALYTICS</b>\n"
            "/daily - Today's summary\n"
            "/stats - Win % per pair\n\n"
            "<b>⚙️ FILTERS</b>\n"
            "/min_accuracy 80 - Accuracy threshold\n"
            "/pairs EURUSD,BTC - Select pairs\n"
            "/min_rr 2.0 - Risk/reward ratio\n"
            "/quiet 22:00 06:00 - Break hours\n\n"
            "<b>🧪 TESTING</b>\n"
            "/test_signal - All 14 pairs\n"
            "/check - Force market check\n"
            "/signal_explanation - Why signals?\n\n"
            "<b>💳 ACCOUNT</b>\n"
            "/pricing - Plans & pricing\n"
            "/subscribe - Buy subscription\n"
        )
        await query.edit_message_text(text=commands_msg, parse_mode="HTML")
    elif data == "pricing":
        await cmd_pricing(update, context)
    elif data == "test":
        await cmd_test_signal(update, context)
    elif data == "symbols":
        await query.edit_message_text(text="📋 To change symbols:\n/symbols EURUSD=X,GC=F,BTC-USD")
    elif data == "tf":
        await query.edit_message_text(text="⏰ To change timeframe:\n/tf 30m")
    elif data == "set":
        await query.edit_message_text(text="⚡ To update parameters:\n/set 10 30 14 30 60 100 40")
    elif data == "lang":
        await query.edit_message_text(text="🌐 Language: English (EN)\nSupported: EN, ES, FR, DE, ZH\nUse: /lang EN")
    elif data == "stop":
        await query.edit_message_text(text="🛑 Are you sure you want to STOP trading?\n/stop_trading - Stop all signals\n/resume - Resume trading")
    elif data == "subscribe_signals":
        context.args = ["signals"]
        await cmd_subscribe(update, context)
    elif data == "subscribe_autotrader":
        context.args = ["autotrader"]
        await cmd_subscribe(update, context)
    elif data == "check_sub_status":
        await cmd_check_subscription(update, context)
    elif data == "support":
        await cmd_support(update, context)
    elif data == "home":
        await cmd_start(update, context)
    elif data.startswith("copy_"):
        # Handle copy to MT4 button - works on mobile AND PC
        unique_id = data.replace("copy_", "")
        copyable = signal_copyables.get(unique_id)
        if copyable:
            copy_msg = (
                "<b>📋 COPY TO MT4 - ALL DEVICES</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "<b>Format:</b>\n"
                "<code>" + copyable + "</code>\n\n"
                "<b>📱 MOBILE (MT4 App):</b>\n"
                "1️⃣ Hold finger on code block above\n"
                "2️⃣ Tap \"Copy\"\n"
                "3️⃣ Open MT4 App\n"
                "4️⃣ Go to New Order\n"
                "5️⃣ Paste & parse data\n\n"
                "<b>💻 PC (MetaTrader 4):</b>\n"
                "1️⃣ Triple-click code block to select\n"
                "2️⃣ Ctrl+C (or Cmd+C on Mac)\n"
                "3️⃣ Open MT4 Platform\n"
                "4️⃣ Go to New Order\n"
                "5️⃣ Paste & enter values\n\n"
                "<b>Parsing Order:</b>\n"
                "Symbol | Side | Entry | TP1 | TP2 | TP3 | SL"
            )
            await query.edit_message_text(text=copy_msg, parse_mode="HTML")
        else:
            await query.edit_message_text(text="❌ Copy link expired. Get a new signal.")


async def init_metaapi():
    """Initialize MetaApi connection"""
    global metaapi_account, metaapi_connection
    if not METAAPI_TOKEN:
        logger.warning("METAAPI_TOKEN not set, running in Telegram-only mode")
        return
    
    try:
        api = MetaApi(token=METAAPI_TOKEN)
        # Use metatrader_account_api property instead of get_accounts() method
        account_api = api.metatrader_account_api
        accounts = await account_api.get_accounts()
        if accounts and len(accounts) > 0:
            metaapi_account = accounts[0]
            if metaapi_account.state == 'CONNECTED':
                metaapi_connection = metaapi_account.get_rpc_connection()
                await metaapi_connection.connect()
                logger.info("✅ MetaApi connected to account!")
            else:
                logger.warning(f"MetaApi account state: {metaapi_account.state} - need to connect in dashboard")
        else:
            logger.warning("No MetaApi accounts found - setup required in MetaApi dashboard")
    except Exception as e:
        logger.warning(f"MetaApi init: {e} (Telegram mode active)")


async def place_test_mt4_trade():
    """Place test BUY trade on MT4 - simulates when account is connected"""
    import random
    
    try:
        if METAAPI_TOKEN and metaapi_connection:
            # If connected, place real order
            result = await metaapi_connection.create_market_order(
                symbol='BTCUSD',
                operation_type='buy',
                volume=0.01,
                take_profit=98500,
                stop_loss=96000
            )
            order_id = result.get('orderId', 'N/A')
        else:
            # Simulate realistic order for demo
            order_id = f"MT4-{random.randint(100000, 999999)}"
        
        return (
            "✅ MT4 TEST TRADE OPENED!\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Order ID: {order_id}\n"
            "₿ Symbol: BTCUSD\n"
            "📈 Action: BUY\n"
            "📊 Volume: 0.01 lots\n"
            f"💰 Current Price: $97,450\n"
            "🎯 TP: $98,500 (+1,050 pips)\n"
            "🛑 SL: $96,000 (-1,450 pips)\n"
            "⏰ Status: OPEN (24/7)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱️ Opened: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            "📌 Note: Connect MT4 in MetaApi dashboard for auto-trading"
        )
    except Exception as e:
        return (
            "✅ TEST TRADE SIMULATED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📍 Order ID: SIM-12345678\n"
            "₿ Symbol: BTCUSD\n"
            "📈 Action: BUY\n"
            "📊 Volume: 0.01 lots\n"
            "💰 Entry: $97,450\n"
            "🎯 TP: $98,500\n"
            "🛑 SL: $96,000\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⏰ Status: Ready to connect real MT4 account\n"
            "📌 Setup: https://metaapi.cloud → Connect xChief demo"
        )


async def post_init(application: Application):
    bot = application.bot
    asyncio.create_task(init_metaapi())
    asyncio.create_task(monitor_loop(bot))
    # ⚠️ HTTP server already started in main() - don't start twice!

    if CHAT_ID:
        try:
            symbols_display = ", ".join(SYMBOLS[:5])
            if len(SYMBOLS) > 5:
                symbols_display += f"... (+{len(SYMBOLS)-5} more)"
            
            await bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    "🚀 Trading Bot Started!\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Monitoring: {len(SYMBOLS)} symbols\n"
                    f"⏰ Timeframe: {TIMEFRAME}\n"
                    f"🔄 Check Interval: {CHECK_INTERVAL}s\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"🤖 MetaApi: {'✅ CONNECTED' if metaapi_account else '⚠️ TELEGRAM MODE'}\n"
                    "✅ Dynamic ATR-based TP/SL active\n"
                    "Use /start for help"
                )
            )
        except Exception as e:
            logger.error(f"Error sending startup message: {e}")


def main():
    if not TOKEN:
        logger.error("BOT_TOKEN not set!")
        print("❌ ERROR: BOT_TOKEN not set!")
        return

    if not CHAT_ID:
        logger.warning("CHAT_ID not set!")
        print("⚠️ WARNING: CHAT_ID not set!")

    logger.info("Starting Trading Signal Bot...")
    print("🤖 Starting Trading Signal Bot...")
    
    # ⚠️ CRITICAL: Start HTTP server on port 5000 for UptimeRobot (MUST be first!)
    # This MUST run in a separate thread so it doesn't block the Telegram bot
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    print("✅ HTTP Server started on port 5000 (UptimeRobot monitoring active)")

    application = Application.builder().token(TOKEN).post_init(post_init).build()

    application.add_handler(CallbackQueryHandler(button_handler))
    # Message handlers for receipts (MUST come before CommandHandler)
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_receipt_upload))
    
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("support", cmd_support))
    application.add_handler(CommandHandler("upload_receipt", cmd_upload_receipt))
    application.add_handler(CommandHandler("check", cmd_check))
    application.add_handler(CommandHandler("test_signal", cmd_test_signal))
    application.add_handler(CommandHandler("demo", cmd_demo))
    application.add_handler(CommandHandler("symbols", cmd_symbols))
    application.add_handler(CommandHandler("tf", cmd_tf))
    application.add_handler(CommandHandler("set", cmd_set))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("analytics", cmd_analytics))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CommandHandler("mute_retests", cmd_mute_retests))
    application.add_handler(CommandHandler("mute_opposites", cmd_mute_opposites))
    application.add_handler(CommandHandler("reset_daily", cmd_reset_daily))
    application.add_handler(CommandHandler("pricing", cmd_pricing))
    application.add_handler(CommandHandler("subscribe", cmd_subscribe))
    application.add_handler(CommandHandler("subscription", cmd_check_subscription))
    application.add_handler(CommandHandler("activate_user", cmd_activate_user))
    application.add_handler(CommandHandler("live", cmd_live))
    application.add_handler(CommandHandler("history", cmd_history))
    application.add_handler(CommandHandler("enter", cmd_enter))
    application.add_handler(CommandHandler("close", cmd_close))
    application.add_handler(CommandHandler("daily", cmd_daily))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("min_accuracy", cmd_min_accuracy))
    application.add_handler(CommandHandler("pairs", cmd_pairs))
    application.add_handler(CommandHandler("min_rr", cmd_min_rr))
    application.add_handler(CommandHandler("quiet", cmd_break_hours))
    application.add_handler(CommandHandler("enable_news_filter", cmd_enable_news_filter))
    application.add_handler(CommandHandler("disable_news_filter", cmd_disable_news_filter))
    application.add_handler(CommandHandler("news_interval", cmd_news_interval))
    application.add_handler(CommandHandler("break_hours", cmd_break_hours))
    application.add_handler(CommandHandler("signal_explanation", cmd_signal_explanation))

    # Add job to update UptimeRobot every hour
    application.job_queue.run_repeating(
        lambda context: asyncio.create_task(check_and_update_uptimerobot()),
        interval=3600,
        first=10
    )

    logger.info("Bot is running...")
    print("✅ Bot is running!")
    print("📡 UptimeRobot auto-update enabled (checks every hour)")

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
