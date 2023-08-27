import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
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
    ldatadf = cryptoobj.load_cryptos(tickers, save=True)
    return ldatadf

def main():

    with st.sidebar:
        crypto_pair = st.selectbox("Which crypto pairs do you want to analyse", crypto_pair_dict.keys())

        server_location = st.selectbox("Is the server in the US?", ("No", "Yes"))

        # Check the selected value
        if server_location == "Yes":
            st.write("The server is physically located in the US")
            server_location = "US"
        elif server_location == "No":
            st.write("The server is not physically located in the US")
            server_location = "not-US"

        time_frames_dict = {"1d": 60 * 24, "4h": 60 * 4, "1h": 60, "30m": 30,
                            "15m": 15, "5m": 5, "3m":3 ,"1m": 1}
        time_frame = st.selectbox("Select time-frame", time_frames_dict.keys())

        starting_dates_wrt_time_frame = {"1d":'1 Mar 2017', 
                                        "4h":'1 Jan 2018', 
                                        "1h":'1 Jan 2020',
                                        "30m":'1 Jan 2021',
                                        "15m":'1 May 2021',
                                        "5m":'1 Jun 2022',
                                        "3m":'1 Jan 2022',
                                        "1m":'1 Jun 2023'
                                        }
        starting_date = starting_dates_wrt_time_frame[time_frame]

    st.title(f"Real-time {time_frame} {crypto_pair_dict[crypto_pair]} Chart")

#    rerun_button = st.button("Run App")

    # Create a search bar for the API key
    api_key = st.text_input("Enter your Binance API Key without being seen", type="password")
    api_secret = st.text_input("Enter your Binance API Secret without being seen", type="password")

    # Check if the API key is provided
    if api_key and api_secret :
        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, "..")
        data_folder = os.path.join(root_dir, "data")
        asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))
        crypto = lc.CryptoData(asset_details, data_folder, market = "spot")
        crypto.set_binance_api_keys(api_key, api_secret, server_location=server_location)
        print(f"server location {server_location}")
        crypto.trade_time_units(dt=60, kline_size=time_frame,starting_date=starting_date)

        tickers = crypto.asset_details["Ticker"]
        tickers = list(tickers[tickers == crypto_pair].values)

        ldata_df = get_dataframe(crypto, tickers)
        crypto_name = crypto_pair_dict[crypto_pair]
        target = "UpDown"
        candles = fb.Candles(crypto_name, target=target)
        candles.set_candles(ldata_df)

        # Get the minimum and maximum dates from the dataframe
        min_date = tu.todatetime(ldata_df.index[0])
        max_date = tu.todatetime(ldata_df.index[-1])
        # Set the initial range values
        default_start_date = tu.todatetime(ldata_df.index[-200])                                     
        default_end_date = max_date

        # Time range slider
        start_date, end_date = st.slider("Select date range", min_value=min_date, max_value=max_date,
                                         value=(default_start_date, default_end_date), format="DD/MM/YY - hh:mm")

        candles.buildfeatures()
        fig_vma = candles.ta_vma_plotly(start_date, end_date)
        fig_full = candles.ta_fullplot_plotly(start_date, end_date)

        chart_placeholder_vma = st.empty()
        chart_placeholder_vma.plotly_chart(fig_vma)

        chart_placeholder_full = st.empty()
        chart_placeholder_full.plotly_chart(fig_full)