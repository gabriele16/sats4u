import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import pyfolio as pf
from pathlib import Path
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import path
import sys
from streamlit import cli as stcli

import loadcrypto as lc
import featbuild as fb


crypto_pair_dict = {
    "BTCUSDT": "Bitcoin",
    "BCHUSDT": "BitcoinCash",
    "BNBUSDT": "BinanceChain",
    "EOSUSDT": "EOS",
    "ETCUSDT": "EthClassic",
    "ETHUSDT": "Ethereum",
    "LTCUSDT": "Litecoin",
    "XMRUSDT": "Monero",
    "TRXUSDT": "Tron",
    "ZECUSDT": "ZCash",
    "XLMUSDT": "Stellar",
    "ADAUSDT": "ADA",
    "IOTAUSDT": "IOTA",
    "MKRUSDT": "Maker",
    "DOGEUSDT": "DogeCoin"
}

@st.cache
def get_dataframe(cryptoobj, tickers):
    ldatadf = cryptoobj.load_cryptos(tickers, save=False)
    return ldatadf

def main():
    st.title("Real-time Chart")
    st.write("Updating every minute...")

    crypto_pair = st.selectbox("Which crypto pairs do you want to analyse", crypto_pair_dict.keys())

    server_location = st.selectbox("Is the server in the US?", ("Yes", "No"))

    # Check the selected value
    if server_location == "Yes":
        st.write("The server is physically located in the US")
        server_location = "US"
    elif server_location == "No":
        st.write("The server is not physically located in the US")
        server_location = "not-US"

    time_frames_dict = {"1d": 60 * 24, "4h": 60 * 4, "1h": 60, "30m": 30,
                        "15m": 15, "5m": 5, "1m": 1}
    time_frame = st.selectbox("Select time-frame", time_frames_dict.keys())

    initial_step = st.number_input('Select how many candle-sticks to visualize up to now',
                                   min_value=10,
                                   max_value=600, value=100, step=1)
    initial_step = -initial_step

    # Create a search bar for the API key
    api_key = st.text_input("Enter your Binance API Key without being seen", type="password")
    api_secret = st.text_input("Enter your Binance API Secret without being seen", type="password")

    # Check if the API key is provided
    if api_key and api_secret:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, "..")
        data_folder = os.path.join(root_dir, "data")
        asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))
        crypto = lc.CryptoData(asset_details, data_folder)
        crypto.set_binance_api_keys(api_key, api_secret, server_location=server_location)
        crypto.trade_time_units(dt=60, kline_size=time_frame, period=time_frames_dict[time_frame],
                                starting_date='1 Mar 2017')

        tickers = crypto.asset_details["Ticker"]
        tickers = list(tickers[tickers == crypto_pair].values)

        ldata_df = get_dataframe(crypto, tickers)
        crypto_name = crypto_pair_dict[crypto_pair]
        target = "UpDown"
        candles = fb.Candles(ldata_df, crypto_name, target=target)
        candles.buildfeatures()
        fig = candles.ta_vma_plotly(in_step=initial_step, last_step=0)

        chart_placeholder = st.empty()
        chart_placeholder.plotly_chart(fig)

        while True:
            ldata_df = get_dataframe(crypto, tickers)
            candles = fb.Candles(ldata_df, crypto_name, target=target)
            candles.buildfeatures()
            fig = candles.ta_vma_plotly(in_step=initial_step, last_step=0)

            # Update the chart placeholder with the new figure
            chart_placeholder.plotly_chart(fig)

            # Wait for 60 seconds before the next update
            time.sleep(60)

if __name__ == '__main__':
    main()