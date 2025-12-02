#!/usr/bin/env python3
"""
ULTIMATE AUTO-RECOVERY SYSTEM
- Ensures bot ALWAYS runs
- Restarts on any failure
- Self-heals automatically
- Never stays down
"""
import os
import subprocess
import time
import requests
import signal
import sys
import logging
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - [AUTO-RECOVERY] - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("auto_recover")

BOT_PROCESS = None
RESTART_COUNT = 0
EMERGENCY_MODE = False

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    logger.info("🛑 Auto-recovery system shutting down (but will respawn!)")
    sys.exit(0)

def is_bot_alive():
    """Check if bot is truly alive - HTTP server AND process check"""
    try:
        # Check if HTTP server responds
        response = requests.get("http://localhost:5000/", timeout=3)
        if response.status_code != 200:
            return False
        
        # Check if main.py process exists
        pids = get_bot_process()
        if not pids or not pids[0]:
            logger.warning("Bot process not found!")
            return False
        
        return True
    except Exception as e:
        logger.debug(f"Health check failed: {e}")
        return False

def get_bot_process():
    """Get bot process PID"""
    result = subprocess.run(
        "ps aux | grep 'python main.py' | grep -v grep | awk '{print $2}'",
        shell=True,
        capture_output=True,
        text=True
    )
    pids = result.stdout.strip().split('\n')
    return [p for p in pids if p]

def kill_all_bots():
    """Kill all existing bot processes"""
    try:
        subprocess.run("pkill -9 -f 'python main.py' 2>/dev/null", shell=True)
        time.sleep(2)
        logger.info("✅ Cleaned up old bot processes")
    except:
        pass

def start_bot():
    """Start the bot"""
    global RESTART_COUNT, EMERGENCY_MODE
    RESTART_COUNT += 1
    
    logger.info(f"🚀 Starting bot (attempt #{RESTART_COUNT})...")
    
    try:
        # Start bot in background
        subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent
        )
        logger.info("✅ Bot start command sent")
        
        # Wait for startup
        time.sleep(5)
        
        # Verify startup
        for attempt in range(10):
            time.sleep(1)
            if is_bot_alive():
                logger.info(f"✅ BOT ONLINE (took {attempt+1}s)")
                EMERGENCY_MODE = False
                return True
        
        logger.warning("⚠️ Bot started but not responding yet - will retry")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        return False

def main():
    """Main auto-recovery loop"""
    global RESTART_COUNT, EMERGENCY_MODE
    
    logger.info("=" * 60)
    logger.info("🛡️  ULTIMATE AUTO-RECOVERY SYSTEM ONLINE")
    logger.info("Bot will NEVER stay down - always recovering...")
    logger.info("=" * 60)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initial startup
    logger.info("🔄 Performing initial setup...")
    kill_all_bots()
    time.sleep(2)
    
    for _ in range(3):
        if start_bot():
            break
        time.sleep(5)
    
    # Main monitoring loop
    consecutive_failures = 0
    
    while True:
        try:
            time.sleep(10)  # Check every 10 seconds (aggressive monitoring)
            
            if not is_bot_alive():
                consecutive_failures += 1
                logger.warning(f"⚠️ Bot not responding (failure #{consecutive_failures})")
                
                if consecutive_failures >= 2:
                    logger.error(f"🚨 CRITICAL: Bot down for {consecutive_failures * 10}s - EMERGENCY RESTART")
                    EMERGENCY_MODE = True
                    kill_all_bots()
                    time.sleep(2)
                    
                    # Emergency restart
                    for attempt in range(5):
                        logger.info(f"🔴 EMERGENCY RESTART #{attempt + 1}...")
                        if start_bot():
                            consecutive_failures = 0
                            break
                        time.sleep(3)
                    else:
                        logger.critical("❌ EMERGENCY RESTART FAILED - sleeping 30s before retry")
                        time.sleep(30)
            else:
                # Bot is healthy
                if consecutive_failures > 0:
                    logger.info(f"✅ Bot recovered after {consecutive_failures} failures")
                consecutive_failures = 0
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown signal received")
            break
        except Exception as e:
            logger.error(f"Auto-recovery error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
