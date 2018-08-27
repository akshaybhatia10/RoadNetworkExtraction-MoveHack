#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install curl
cd /tmp
curl -O https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
chmod 777 Anaconda3-4.3.1-Linux-x86_64.sh
./Anaconda3-4.3.1-Linux-x86_64.sh
source ~/.bashrc
conda install -c intel openmp 
export PATH=/home/ubuntu/anaconda3/bin:$PATH:/bin:/usr/bin:/usr/local/bin:/sbin:/usr/sbin
cd ~