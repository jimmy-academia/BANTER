#!/bin/bash

wget https://github.com/jimmy-academia/BANTER/releases/download/data/NFT_data.zip
unzip NFT_data.zip
rm NFT_data.zip
mv NFT_data/* ./
rm -d NFT_data
