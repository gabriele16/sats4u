import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import pyfolio as pf
from pathlib import Path
import csv
import os
import json
import sys
import math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import path
import sys
from streamlit import cli as stcli

import loadcrypto as lc
import featbuild as fb


def main():

    # Create a search bar for the API key
    api_key = st.text_input("Enter your Binance API Key without being seen")
    api_secret = st.text_input("Enter your Binance API Key without being seen")

    # Check if the API key is provided
    if api_key and api_secret:
        # Use the API key to download live prices from Binance
        # Your code to download live prices using the API key goes here
        st.write(f"Using API key 1: {api_key}")
        st.write(f"Using API secret: {api_secret}")

    initial_step = -170
    final_step = -1
    st.title("Real-time Chart")
    st.write("Updating every 15 minutes...")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, "..")
    data_folder = os.path.join(root_dir, "data") 
    asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))
#    secret_data_folder = '/Users/gabrieletocci/Google Drive/My Drive/Colab Notebooks/crypto_project/crypto_data'
    #secret_data_folder = os.path.join(root_dir, "crypto_data")
#    secrets_filename = os.path.join(secret_data_folder, "data.json")
    crypto = lc.CryptoData(asset_details,data_folder)
    crypto.set_binance_api_keys( api_key, api_secret)
#    crypto.load_binance_client(secrets_filename,data1_str = 'DATA1',data2_str = 'DATA2i')
    crypto.trade_time_units(dt=60,kline_size="1d",period=60*24,starting_date = '1 Mar 2017')

    tickers=crypto.asset_details["Ticker"]
    tickers = list(tickers[tickers=='BTCUSDT'].values)
    ldata_df = crypto.load_cryptos(tickers,save = True)
    crypto_name = "Bitcoin"
    target = "UpDown"
    crypto = fb.Candles(ldata_df,crypto_name, target = target)
    crypto.buildfeatures()

    fig = crypto.ta_vma_plotly(in_step=initial_step, last_step=final_step)

    while True:
        # Retrieve new data and update the chart
        # Your code for retrieving new data and updating the chart here
        
        # Update the chart with the new data
        # Modify the 'fig' object accordingly
        
        st.plotly_chart(fig)

        # Wait for 15 minutes before the next update
        time.sleep(15 * 60)    

# Run the streamlit app
# streamlit run src/sats4ulive.py    
if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

