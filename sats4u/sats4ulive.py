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
from . import loadcrypto as lc
from . import featbuild as fb
from . import timeutils as tu

crypto_pair_dict = {
    "BTCUSDT": "Bitcoin",
    "BCHUSDT": "Bitcoin Cash",
    "BNBUSDT": "Binance Coin",
    "EOSUSDT": "EOS.IO",
    "ETCUSDT": "Ethereum Classic",
    "ETHUSDT": "Ethereum",
    "LTCUSDT": "Litecoin",
    "XMRUSDT": "Monero",
    "TRXUSDT": "TRON",
    "XLMUSDT": "Stellar",
    "ADAUSDT": "Cardano",
    "IOTAUSDT": "IOTA",
    "MKRUSDT": "Maker",
    "DOGEUSDT": "Dogecoin"
}

@st.cache
def get_dataframe(cryptoobj, tickers):
    ldatadf = cryptoobj.load_cryptos(tickers, save=False)
    return ldatadf

def main():

    with st.sidebar:
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
        update_interval = time_frames_dict[time_frame] * 60/2 

    st.title(f"Real-time {time_frame} {crypto_pair_dict[crypto_pair]} Chart")
    rerun_button = st.button("Run App")

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
        print(f"server location {server_location}")
        crypto.trade_time_units(dt=60, kline_size=time_frame, period=time_frames_dict[time_frame],
                                starting_date='1 Mar 2017')

        tickers = crypto.asset_details["Ticker"]
        tickers = list(tickers[tickers == crypto_pair].values)

        ldata_df = get_dataframe(crypto, tickers) 
        crypto_name = crypto_pair_dict[crypto_pair]
        target = "UpDown"
        candles = fb.Candles(ldata_df, crypto_name, target=target)

        # Get the minimum and maximum dates from the dataframe
        min_date = tu.todatetime(ldata_df.index.min()).date()
        max_date = tu.todatetime(ldata_df.index.max()).date()

        # Set the initial range values
        default_start_date = tu.todatetime(ldata_df.index[-200]).date()
        default_end_date = tu.todatetime(ldata_df.index[-1]).date()

        # Time range slider
        start_date, end_date = st.slider("Select date range", min_value=min_date, max_value=max_date,
                                         value=(default_start_date, default_end_date))
        
        candles.buildfeatures()
        fig_vma = candles.ta_vma_plotly(start_date, end_date)
        fig_full = candles.ta_fullplot_plotly(start_date, end_date)

        chart_placeholder_vma = st.empty()
        chart_placeholder_vma.plotly_chart(fig_vma)

        chart_placeholder_full = st.empty()
        chart_placeholder_full.plotly_chart(fig_full)

        # Timer variables
        last_update_time = time.time()

        iteration = 0

        while rerun_button or time.time() - last_update_time <= update_interval:
            last_update_time = time.time()

            iteration += 1

            ### Test code
            crypto.set_binance_api_keys(api_key, api_secret, server_location=server_location)
            print(f"server location {server_location}")
            crypto.trade_time_units(dt=60, kline_size=time_frame, period=time_frames_dict[time_frame],
                                    starting_date='1 Mar 2017')     
            ### End test code       

            crypto_name = crypto_pair_dict[crypto_pair]
            candles = fb.Candles(ldata_df, crypto_name, target=target)
            candles.buildfeatures()
            fig_vma = candles.ta_vma_plotly(start_date, end_date)
            fig_full = candles.ta_fullplot_plotly(start_date, end_date)

            chart_placeholder_vma.plotly_chart(fig_vma)
            chart_placeholder_full.plotly_chart(fig_full)