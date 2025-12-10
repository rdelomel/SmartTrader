#!/bin/bash
# Setup script for DigitalOcean Droplet deployment

set -e

echo "Setting up SmartTrader on DigitalOcean..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Create application directory
sudo mkdir -p /opt/smarttrader
sudo chown $USER:$USER /opt/smarttrader
cd /opt/smarttrader

# Clone repository (or copy files)
# git clone <your-repo-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs models data

# Create systemd service file
sudo tee /etc/systemd/system/smarttrader.service > /dev/null <<EOF
[Unit]
Description=SmartTrader Trading Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/smarttrader
Environment="PATH=/opt/smarttrader/venv/bin"
ExecStart=/opt/smarttrader/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable smarttrader
sudo systemctl start smarttrader

# Setup firewall (if needed)
sudo ufw allow 8000/tcp  # Dashboard port
sudo ufw enable

echo "SmartTrader setup complete!"
echo "Check status with: sudo systemctl status smarttrader"
echo "View logs with: sudo journalctl -u smarttrader -f"

