#!/usr/bin/env bash

python3 -m spacy download en_core_web_md

# install hdbscan for BERTopic
sudo apt-get update
sudo apt-get install python3-dev

# requirements
pip install --upgrade pip
pip install --upgrade nbformat
python3 -m pip install -r requirements.txt

python3 src/emotions.py --filepath ./data/fake_or_real_news.csv --column title


# Find out why the plots are with color.