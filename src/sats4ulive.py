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

    server_location = st.selectbox("Is the server in the US?", ("Yes", "No"))
    crypto_pair = st.selectbox("Which crypto pairs do you want to analyse", ("BTCUSDT", "ETHUSDT", "LTCUSDT"))

    crypto_pair_dict = {"BTCUSDT": "Bitcoin", "ETHUSDT": "Ethereum", "LTCUSDT": "Litecoin"}

    # Check the selected value
    if server_location == "Yes":
        st.write("The server is physically located in the US")
        server_location = "US"
    elif server_location == "No":
        st.write("The server is not physically located in the US")
        server_location = "not-US"
    
    # Create a search bar for the API key
    api_key = st.text_input("Enter your Binance API Key without being seen", type = "password")
    api_secret = st.text_input("Enter your Binance API Secret without being seen", type = "password")

    # Check if the API key is provided
    if api_key and api_secret:
        # Use the API key to download live prices from Binance
        # Your code to download live prices using the API key goes here
        # st.write(f"Using API key : {api_key}")
        # st.write(f"Using API secret: {api_secret}")

        initial_step = -170
        final_step = -1
        st.title("Real-time Chart")
        st.write("Updating every 15 minutes...")

        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, "..")
        data_folder = os.path.join(root_dir, "data") 
        asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))
        crypto = lc.CryptoData(asset_details,data_folder)
        crypto.set_binance_api_keys( api_key, api_secret, server_location = server_location)
        crypto.trade_time_units(dt=60,kline_size="1d",period=60*24,starting_date = '1 Mar 2017')

        tickers=crypto.asset_details["Ticker"]
        tickers = list(tickers[tickers==crypto_pair].values)
        ldata_df = crypto.load_cryptos(tickers,save = False)
        crypto_name = crypto_pair_dict[crypto_pair]
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

