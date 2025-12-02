#!/usr/bin/env python3
"""
Lightweight Uptime Monitor for Trading Signal Bot
Checks bot health every 20 seconds and sends Telegram alerts
"""

import os
import time
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BOT_URL = "http://localhost:5000/"
CHECK_INTERVAL = 20  # seconds

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Track bot status
bot_status = {"is_up": None, "last_down_time": None, "down_count": 0}


def send_alert(message):
    """Send Telegram alert"""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning(f"Skipping alert (no token/chat): {message}")
        return
    
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=data, timeout=5)
        logger.info(f"Alert sent: {message}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


def check_bot_health():
    """Check if bot HTTP server is responding"""
    try:
        response = requests.get(BOT_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def monitor_bot():
    """Main monitoring loop"""
    logger.info("🔍 Uptime Monitor Started")
    logger.info(f"Checking {BOT_URL} every {CHECK_INTERVAL} seconds")
    send_alert(f"🟢 <b>Uptime Monitor Active</b>\nChecking bot every {CHECK_INTERVAL}s\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        try:
            is_up = check_bot_health()
            
            # BOT WENT DOWN
            if not is_up and bot_status["is_up"] is not True:
                bot_status["down_count"] += 1
                if bot_status["down_count"] == 1:  # First failure
                    logger.warning("⚠️ Bot health check failed (1st attempt)")
                elif bot_status["down_count"] == 3:  # After 3 failures (60 seconds)
                    bot_status["is_up"] = False
                    bot_status["last_down_time"] = datetime.now()
                    alert = (
                        "🔴 <b>BOT DOWN!</b>\n"
                        f"Status: Cannot reach {BOT_URL}\n"
                        f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                        "⚡ UptimeRobot should auto-restart..."
                    )
                    send_alert(alert)
                    logger.error(alert)
            
            # BOT CAME BACK UP
            elif is_up and bot_status["is_up"] is not True:
                downtime = None
                if bot_status["last_down_time"]:
                    downtime = (datetime.now() - bot_status["last_down_time"]).total_seconds()
                
                bot_status["is_up"] = True
                bot_status["down_count"] = 0
                
                alert = (
                    "🟢 <b>BOT BACK ONLINE!</b>\n"
                    f"Status: ✅ Responding\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                if downtime:
                    alert += f"\nDowntime: {int(downtime)} seconds"
                send_alert(alert)
                logger.info(f"✅ Bot is back online (downtime: {int(downtime)}s)" if downtime else "✅ Bot is online")
            
            # BOT STILL UP - LOG STATUS
            elif is_up and bot_status["is_up"] is True:
                logger.info(f"✅ Bot healthy - {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("❌ ERROR: BOT_TOKEN or CHAT_ID not set!")
        print("❌ ERROR: BOT_TOKEN or CHAT_ID not set!")
        exit(1)
    
    monitor_bot()
