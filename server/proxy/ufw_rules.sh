
# reset any existing rules 
sudo ufw --force reset

# localhost  -> nginx
sudo ufw allow from 127.0.0.1/24 to any port 80
sudo ufw allow from 127.0.0.1/24 to any port 8443

# home network
sudo ufw allow from 192.168.1.0/24 to any port 443 # wazuh
sudo ufw allow from 192.168.1.0/24 to any port 80 # nginx

# ssh 
sudo ufw allow 22/tcp

# wazuh
# sudo ufw allow 1515/tcp
# sudo ufw allow 1514/tcp
sudo ufw allow from 192.168.1.0/24 to any port 443


sudo ufw status numbered
