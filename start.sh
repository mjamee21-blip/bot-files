#!/bin/bash

# Trading Signal Bot - Startup Script
# Starts all three bot processes

set -e

echo "🤖 Starting Trading Signal Bot..."
echo "=================================="

# Check Python
python3 --version

# Check .env file exists
if [ ! -f .env ]; then
    echo "❌ ERROR: .env file not found!"
    echo "Please create .env from .env.example"
    echo "  cp .env.example .env"
    echo "  nano .env  # Edit with your API keys"
    exit 1
fi

echo "✅ .env file found"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🚀 Starting bot processes..."
echo "=================================="

# Start main bot
echo "Starting main.py (Trading Bot)..."
python3 main.py > logs/main.log 2>&1 &
MAIN_PID=$!
echo "✅ Main bot started (PID: $MAIN_PID)"

# Start auto-recovery
echo "Starting auto_recover.py (Auto-Recovery)..."
python3 auto_recover.py > logs/auto_recover.log 2>&1 &
RECOVER_PID=$!
echo "✅ Auto-recovery started (PID: $RECOVER_PID)"

# Start uptime monitor
echo "Starting uptime_monitor.py (Uptime Monitor)..."
python3 uptime_monitor.py > logs/uptime_monitor.log 2>&1 &
MONITOR_PID=$!
echo "✅ Uptime monitor started (PID: $MONITOR_PID)"

echo ""
echo "=================================="
echo "✅ ALL PROCESSES STARTED!"
echo "=================================="
echo ""
echo "Process IDs:"
echo "  Main Bot:       $MAIN_PID"
echo "  Auto-Recovery:  $RECOVER_PID"
echo "  Uptime Monitor: $MONITOR_PID"
echo ""
echo "To view logs:"
echo "  tail -f logs/main.log"
echo "  tail -f logs/auto_recover.log"
echo "  tail -f logs/uptime_monitor.log"
echo ""
echo "To stop all processes:"
echo "  kill $MAIN_PID $RECOVER_PID $MONITOR_PID"
echo ""

# Keep script running
wait
