# 🚀 DEPLOYMENT GUIDE

## Download & Setup Steps

### 1. DOWNLOAD ALL FILES

Download the `bot-files` folder containing:
- `main.py` - Main trading bot (3522 lines)
- `auto_recover.py` - Auto-recovery system
- `uptime_monitor.py` - Uptime monitoring
- `requirements.txt` - Python dependencies
- `.env.example` - Environment template
- `start.sh` - Startup script
- `README.md` - Full documentation

### 2. UPLOAD TO YOUR SERVER

```bash
# Option A: Using SCP
scp -r bot-files user@your-server.com:/home/user/

# Option B: Using SFTP (FileZilla, WinSCP)
# Upload bot-files folder to your home directory

# Option C: Git Clone
# If you push to GitHub/GitLab
git clone https://github.com/yourname/trading-bot.git
cd trading-bot
```

### 3. SSH INTO SERVER

```bash
ssh user@your-server.com
cd bot-files
```

### 4. SETUP ENVIRONMENT

```bash
# Copy env template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Fields in .env:**
```
BOT_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
FRED_API_KEY=your_fred_api_key
FINNHUB_API_KEY=your_finnhub_api_key
USDT_WALLET=TYr7MXeg8AyqsWwwDe7PrEfN7zdrB3VcSt
```

### 5. INSTALL DEPENDENCIES

```bash
# Update package manager
sudo apt update && sudo apt upgrade

# Install Python if needed
sudo apt install python3 python3-pip

# Install bot dependencies
pip install -r requirements.txt
```

### 6. CREATE LOGS DIRECTORY

```bash
mkdir -p logs
chmod 755 logs
```

### 7. RUN THE BOT

**Option A: Simple Start (foreground)**
```bash
python3 main.py
```

**Option B: Background with start.sh**
```bash
bash start.sh
```

**Option C: Using screen (detachable)**
```bash
screen -S trading-bot
python3 main.py
# Press Ctrl+A then D to detach
# screen -r trading-bot  # to reattach
```

**Option D: Using nohup (no hangup)**
```bash
nohup python3 main.py > logs/bot.log 2>&1 &
nohup python3 auto_recover.py > logs/recover.log 2>&1 &
nohup python3 uptime_monitor.py > logs/uptime.log 2>&1 &
```

### 8. VERIFY BOT IS RUNNING

```bash
# Check processes
ps aux | grep python3

# Check logs
tail -f logs/main.log

# Test Telegram
# Send /start to your bot on Telegram
```

---

## PRODUCTION SETUP (Recommended)

### Using PM2 (Node.js Process Manager)

```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install PM2 globally
sudo npm install -g pm2

# Create PM2 config
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'trading-bot',
      script: 'main.py',
      interpreter: 'python3',
      instances: 1,
      autorestart: true,
      max_restarts: 10,
      error_file: 'logs/error.log',
      out_file: 'logs/out.log'
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

# Start PM2
pm2 start ecosystem.config.js

# Monitor
pm2 monit

# Make it persistent across reboots
pm2 startup
pm2 save
```

### Using Systemd (Linux Native)

```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service
```

Paste this:
```ini
[Unit]
Description=Trading Signal Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bot-files
ExecStart=/usr/bin/python3 /home/ubuntu/bot-files/main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/bot-files/logs/systemd.log
StandardError=append:/home/ubuntu/bot-files/logs/systemd-error.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

---

## SERVER RECOMMENDATIONS

**Minimum:**
- RAM: 1GB
- CPU: 1 Core
- Disk: 500MB

**Recommended:**
- RAM: 2-4GB
- CPU: 2-4 Cores
- Disk: 10GB

**Hosting Suggestions:**
- AWS EC2 (t3.micro = free tier)
- DigitalOcean ($4/month)
- Linode ($5/month)
- Vultr ($2.50/month)
- Heroku ($7/month - easiest)

---

## MONITORING & MAINTENANCE

### Check Bot Status
```bash
# Process check
ps aux | grep python3

# Port check (if using HTTP server)
netstat -tuln | grep 5000

# Memory usage
free -h

# Disk space
df -h
```

### View Logs
```bash
# Main bot
tail -f logs/main.log

# Auto-recovery
tail -f logs/auto_recover.log

# Uptime monitor
tail -f logs/uptime_monitor.log

# All logs
tail -f logs/*.log
```

### Restart Bot
```bash
# Kill all processes
pkill -f "python3 main.py"
pkill -f "python3 auto_recover.py"
pkill -f "python3 uptime_monitor.py"

# Wait 2 seconds
sleep 2

# Restart
bash start.sh
```

### Update Code
```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies
pip install -r requirements.txt

# Restart bot
pkill -f python3
sleep 2
bash start.sh
```

---

## TROUBLESHOOTING

### Bot Not Starting
```bash
# Check Python version
python3 --version

# Check .env file
cat .env

# Check dependencies
pip list

# Try running directly
python3 main.py
```

### Module Not Found
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt

# Force reinstall
pip install --force-reinstall -r requirements.txt
```

### Memory Error
```bash
# Check memory
free -h

# Kill zombie processes
pkill -9 python3

# Reduce resource usage
# (optional) Adjust CHECK_INTERVAL in main.py
```

### Telegram Not Connecting
```bash
# Test bot token
curl -s https://api.telegram.org/bot{YOUR_BOT_TOKEN}/getMe

# Test chat ID
curl -s -X POST https://api.telegram.org/bot{YOUR_BOT_TOKEN}/sendMessage \
  -d chat_id=YOUR_CHAT_ID \
  -d text="Test message"
```

---

## SECURITY CHECKLIST

- [ ] Created unique `.env` file (never share!)
- [ ] Used strong API keys/tokens
- [ ] Limited server access (SSH key only, no password)
- [ ] Disabled root login
- [ ] Setup firewall rules
- [ ] Regular backups of trading data
- [ ] Monitor bot logs regularly
- [ ] Update packages monthly

---

## GETTING HELP

1. Check `/signal_explanation` in Telegram bot
2. Review logs: `tail -f logs/*.log`
3. Verify environment: `cat .env`
4. Test API keys manually
5. Contact support via bot: `/support`

---

**Status**: Ready to Deploy ✅  
**Last Updated**: December 2, 2025
