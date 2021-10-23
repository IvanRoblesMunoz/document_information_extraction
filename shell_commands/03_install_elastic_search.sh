# Reference: https://www.how2shout.com/linux/how-to-install-elasticsearch-on-ubuntu-20-04-lts-easy-steps/

# 1. Install java if not there already
#sudo apt install default-java
#java --version

# 2. add GPG key
sudo apt-get install apt-transport-https
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# 3. Add elasticsearch to debian repository
sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'

# 4. update
sudo apt update

# 5. install
sudo apt install elasticsearch


# 6. Commands 
# 6.1 enable
sudo systemctl enable elasticsearch

# 6.2 Start
# sudo systemctl start elasticsearch

# 6.3 Check status
# sudo systemctl status elasticsearch

# 6.4 stop
# sudo systemctl stop elasticsearch

# 6.5 Test by checking port number
# curl -X GET "localhost:9200/"

# 7 Remove
# sudo apt-get --purge autoremove elasticsearch
# sudo rm -rf /var/lib/elasticsearch/
# sudo rm -rf /etc/elasticsearch