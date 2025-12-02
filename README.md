# 🤖 Telegram Trading Signal Bot - Complete Package

**Professional Python-based Telegram bot** delivering automated forex & crypto trading signals using pivot points + ichimoku confirmation across 14 trading pairs.

## 📦 PACKAGE CONTENTS

```
bot-files/
├── main.py                 # Main bot (3522 lines) - ALL FEATURES
├── auto_recover.py         # Auto-recovery system (never goes down)
├── uptime_monitor.py       # Health monitoring + Telegram alerts
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

---

## ✨ KEY FEATURES

✅ **14 Trading Pairs**: 12 Forex + XAUUSD (Gold) + BTC-USD (Bitcoin)
✅ **Pure Pivot Point Breakouts**: With ichimoku confirmation  
✅ **Technical Analysis Charts**: Auto-generated with entry/exit levels
✅ **Manual Trade Tracking**: `/enter` and `/close` commands
✅ **User-Configurable Filters**: Accuracy, pairs, R/R, break hours
✅ **USDT Payment Plans**: $5 Signals / $10 Autotrader (TRC20)
✅ **Payment Receipt Upload**: Photo/document verification
✅ **Live Support Chat**: Ticket system with admin responses
✅ **Daily Analytics**: Win rate, pips, best pairs, hourly stats
✅ **Auto-Recovery**: Never stays down - self-heals automatically
✅ **Uptime Monitoring**: Health checks + Telegram alerts
✅ **24/7 Operation**: Bulletproof raw socket HTTP server

---

## 🚀 QUICK START

### 1. **Setup on Linux/Ubuntu Server**

```bash
# Clone or download files to your server
mkdir -p ~/trading-bot
cd ~/trading-bot

# Copy all bot-files here
# ... (copy main.py, auto_recover.py, etc.)

# Install Python 3.9+
python3 --version

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 2. **Add Environment Variables**

Edit `.env` and fill in:
- `BOT_TOKEN` - Get from BotFather on Telegram
- `CHAT_ID` - Your Telegram chat ID
- `METAAPI_TOKEN` - For MT4 integration (optional)
- `FRED_API_KEY` - Free US economic data (register: https://fred.stlouisfed.org)
- `FINNHUB_API_KEY` - Market data (free: https://finnhub.io)
- `UPTIMEROBOT_API_KEY` - Uptime monitoring (optional)
- `CRONJOB_API_KEY` - Cron-job.org keep-alive (optional)
- `USDT_WALLET` - Your TRC20 USDT wallet address

### 3. **Run the Bot**

```bash
# Start all three processes in the background

# Terminal 1: Main Bot
python3 main.py &

# Terminal 2: Auto-Recovery Guardian
python3 auto_recover.py &

# Terminal 3: Uptime Monitor
python3 uptime_monitor.py &

# Or use a process manager (recommended for production)
# Using supervisor/systemd/pm2
```

### 4. **Using PM2 (Recommended)**

```bash
# Install PM2 globally
npm install -g pm2

# Create PM2 config (ecosystem.config.js)
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'trading-bot',
      script: 'main.py',
      interpreter: 'python3',
      instances: 1,
      autorestart: true,
      watch: false
    },
    {
      name: 'auto-recovery',
      script: 'auto_recover.py',
      interpreter: 'python3',
      instances: 1,
      autorestart: true
    },
    {
      name: 'uptime-monitor',
      script: 'uptime_monitor.py',
      interpreter: 'python3',
      instances: 1,
      autorestart: true
    }
  ]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js

# Check status
pm2 status

# View logs
pm2 logs
```

### 5. **Using Systemd (Alternative)**

```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service
```

```ini
[Unit]
Description=Trading Signal Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/your_user/trading-bot
ExecStart=/usr/bin/python3 /home/your_user/trading-bot/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
sudo journalctl -u trading-bot -f
```

---

## 📋 COMMANDS

| Command | Purpose | Example |
|---------|---------|---------|
| `/history` | Last 10 signals | View recent signals |
| `/enter` | Record trade entry | `/enter EURUSD 1.08750 BUY` |
| `/close` | Close & calc P&L | `/close EURUSD 1.08900` |
| `/daily` | Today's stats | 8 signals today |
| `/stats` | Pair performance | Win rate per pair |
| `/min_accuracy` | Accuracy filter | `/min_accuracy 80` |
| `/pairs` | Select pairs | `/pairs EURUSD,GBPUSD,XAUUSD` |
| `/min_rr` | R/R filter | `/min_rr 2.0` |
| `/quiet` | Break hours | `/quiet 22:00 06:00` |
| `/signal_explanation` | Why signals generate | Shows all active filters |
| `/subscribe` | Subscribe plan | Signal ($5) or Autotrader ($10) |
| `/upload_receipt` | Payment receipt | Verify USDT payment |
| `/support` | Live support | Chat with admin |
| `/test_signal` | Test all pairs | Manual signal check |
| `/pricing` | View plans | Subscription tiers |

---

## 🔧 SYSTEM REQUIREMENTS

- **Python**: 3.9 or higher
- **RAM**: 1GB minimum (2GB recommended)
- **Disk**: 500MB minimum
- **Network**: Stable internet connection
- **OS**: Linux/Ubuntu (or Windows/Mac with Python)

---

## 📊 ARCHITECTURE

### Three-Process System

1. **main.py** - Main Trading Bot
   - Fetches market data every 30 minutes
   - Analyzes 14 trading pairs
   - Sends signals to users
   - Handles Telegram commands
   - Tracks manual trades
   - Generates technical charts

2. **auto_recover.py** - Auto-Recovery Guardian
   - Monitors main.py continuously
   - Auto-restarts on failure
   - Checks every 10 seconds
   - Emergency restart mode
   - Zero downtime guarantee

3. **uptime_monitor.py** - Uptime Monitor
   - Health checks every 20 seconds
   - Sends Telegram alerts
   - Tracks downtime
   - Recovery notifications

---

## 🛡️ MONITORING & ALERTS

### UptimeRobot Integration
```
UptimeRobot → Monitors HTTP server
            → Auto-restarts on downtime
            → Status: Active ✅
```

### Cron-Job.org Integration
```
Cron-job.org → Keep-alive pings every 5 mins
             → Prevents idle timeout
             → Status: Active ✅
```

### Telegram Alerts
```
Bot sends alerts on:
- Status changes (UP/DOWN)
- Recovery events
- Error conditions
- Daily stats
```

---

## 📈 TRADING LOGIC

### Signal Generation

**When to SEND signals:**
1. Price breaks pivot point (support/resistance)
2. Ichimoku confirms the breakout
3. User's filters pass (accuracy, R/R, break hours)
4. Market conditions optimal (liquidity, volatility)

**Entry Signal Includes:**
- Entry price
- TP1, TP2, TP3 (profit targets)
- SL (stop loss)
- Copyable MT4 format
- Technical analysis chart

**Exit Notifications:**
- Auto-sent when TP1/TP2/TP3/SL hit
- No manual checking needed
- Includes exit price and pips

---

## 💳 MONETIZATION

### Subscription Plans

| Feature | Free | Signals | Autotrader |
|---------|------|---------|-----------|
| Price | Free | $5/month | $10/month |
| Signals/day | 1 | 10 | ∞ |
| Trading pairs | 5 | 14 | 14 |
| Auto-trading | ❌ | ❌ | ✅ |
| Payment | - | USDT | USDT |

### Payment Flow

1. User clicks `/subscribe signals` or `/subscribe autotrader`
2. Bot sends USDT wallet address
3. User sends payment on TRC20
4. User uploads receipt with `/upload_receipt`
5. Admin verifies payment
6. Admin activates with `/activate_user {user_id} {plan}`
7. User gets full access (24h activation)

---

## 🐛 TROUBLESHOOTING

### Bot Not Responding
```bash
# Check if process is running
ps aux | grep python3 | grep main.py

# Check logs
tail -f nohup.out
journalctl -u trading-bot -f  # if using systemd

# Restart manually
python3 main.py &
```

### API Errors
```bash
# Check environment variables
cat .env

# Verify tokens
echo $BOT_TOKEN
echo $FRED_API_KEY

# Test API connectivity
curl -s https://api.telegram.org/bot{BOT_TOKEN}/getMe
```

### Memory Issues
```bash
# Monitor memory usage
free -h
top  # then press 'q' to quit

# Kill zombie processes
pkill -9 python3  # CAUTION: kills all Python processes!
```

---

## 📝 FILE STRUCTURE

```
trading-bot/
├── main.py                 # Main bot code (3522 lines)
├── auto_recover.py         # Recovery system (174 lines)
├── uptime_monitor.py       # Monitoring system (119 lines)
├── requirements.txt        # Dependencies
├── .env                    # Your secrets (NEVER share!)
├── .env.example           # Template (safe to share)
├── ecosystem.config.js    # PM2 config (if using PM2)
└── README.md              # This file
```

---

## 🔐 SECURITY NOTES

⚠️ **NEVER share your .env file!**
- Contains API keys and secrets
- Git-ignore .env automatically
- Only share .env.example

✅ **Best Practices:**
- Use environment variables for all secrets
- Rotate API keys regularly
- Use VPN for server access
- Monitor bot access logs
- Keep Python updated

---

## 📞 SUPPORT

For issues, questions, or customizations:
- Check `/signal_explanation` in bot
- Use `/support` for live chat
- Review logs: `pm2 logs` or `journalctl -u trading-bot -f`
- Check environment: `env | grep TRADING`

---

## 📄 LICENSE

This bot is proprietary. Do not modify or redistribute without permission.

---

**Version**: 1.0.0  
**Last Updated**: December 2, 2025  
**Status**: ✅ Production Ready - 24/7 Operation Tested
