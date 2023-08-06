import numpy as np
import pandas as pd
import math
import os
import datetime
import time
from datetime import datetime, timedelta
from . import timeutils as tu
from binance import BinanceSocketManager
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException  # here
from binance.client import Client
from dateutil import parser
from pathlib import Path
from tqdm import tqdm
import traceback
import json


class CryptoData:
    def __init__(self, asset_details, data_folder, verbose=True, market = "spot"):

        self.asset_details = asset_details
        self.data_folder = data_folder
        self.verbose = verbose
        self.market = market

        self.tick_to_id = {
            "BTCUSDT": 1,
            "BCHUSDT": 2,
            "BNBUSDT": 0,
            "EOSUSDT": 5,
            "ETCUSDT": 7,
            "ETHUSDT": 6,
            "LTCUSDT": 9,
            "XMRUSDT": 11,
            "TRXUSDT": 13,
            "XLMUSDT": 12,
            "ADAUSDT": 3,
            "IOTAUSDT": 8,
            "MKRUSDT": 10,
            "DOGEUSDT": 4,
        }

        self.crypto_pair_dict = {
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

        self.tick_to_id_df = pd.DataFrame(
            np.array([list(self.tick_to_id.keys()), list(
                self.tick_to_id.values())]).T, columns=["Ticker", "Asset_ID"]
        )
        self.tick_to_id_df["Asset_ID"] = self.tick_to_id_df["Asset_ID"].astype(
            int)
        self.asset_details = pd.merge(
            self.asset_details, self.tick_to_id_df, on="Asset_ID")

    def _binance_api_constants(self):

        self._binsizes = {"1m": 1, "5m": 5, "15m": 15, 
                          "30m":30 ,"1h": 60, "4h":60*4 ,
                          "1d": 60*24}
        self._batch_size = 750

    def trade_time_units(self, dt=60, kline_size="15m", starting_date="1 Mar 2022"):

        self._binance_api_constants()
        self.dt = dt
        self.kline_size = kline_size
        self.period = self._binsizes[kline_size]
        self.starting_date = starting_date
        self.dt_daily = 60 * 60 * 24
        self.minute = 1 * self.dt
        self.hour = 60 * self.minute
        self.day = self.hour * 24
        self.week = self.day * 7
        self.month = self.day * 30
        self.year = self.day * 365

    def load_binance_client(self, secrets_filename, data1_str="DATA1", data2_str="DATA2i", testnet=False):

        # secrets_filename = data_folder+'crypto_data/data.json'
        api_keys = {}
        with open(secrets_filename, "r") as f:
            api_keys = json.loads(f.read())

        # API
        # Enter your own API-key here
        self._binance_api_key = api_keys[data1_str]
        # Enter your own API-secret here
        self._binance_api_secret = api_keys[data2_str]
        self.binance_client = Client(
            self._binance_api_key, self._binance_api_secret, testnet=testnet)
    
    def set_binance_api_keys(self, api_key, api_secret, server_location = 'not-US', testnet=False):
        self._binance_api_key = api_key
        self._binance_api_secret = api_secret
        if server_location == 'US':
            self.binance_client = Client(
                self._binance_api_key, self._binance_api_secret, tld='us', testnet=testnet)
            print("Extracting data from Binance.us")            
        elif server_location == 'not-US':
            self.binance_client = Client(
                self._binance_api_key, self._binance_api_secret, tld='com',testnet=testnet)
            print("Extracting data from Binance.com")
        else:
            raise ValueError('server_location must be either US or not-US')

    # FUNCTIONS
    def _minutes_of_new_data(self, symbol, data, source):

        if len(data) > 0:
            old = parser.parse(data["Date"].iloc[-1])
        elif source == "binance":
            old = datetime.strptime(self.starting_date, "%d %b %Y")
        if source == "binance":
            try:
                new = self.get_new_spot_or_futures_klines(symbol, self.kline_size, market=self.market)
                
            except BinanceAPIException as e:
                print(e)
                print(
                    "Something went wrong. Error occured at %s. Wait for 1 hour."
                    % (datetime.datetime.now().astimezone(datetime.timezone("UTC")))
                )
                time.sleep(3600)
                self.binance_client = Client(
                    self._binance_api_key, self._binance_api_secret)
                new = self.get_new_spot_or_futures_klines(symbol, self.kline_size, market=self.market)
 
        return old, new

    def get_new_spot_or_futures_klines(self, symbol, kline_size, market="spot"):
        
        if market == "spot":
            new_klines = pd.to_datetime(
                self.binance_client.get_klines(
                symbol = symbol, interval = kline_size)[-1][0], unit="ms")

        elif market == "futures":
            new_klines = pd.to_datetime(
                self.binance_client.futures_klines(
                symbol = symbol, interval = kline_size)[-1][0], unit="ms")
        
        return new_klines

    def get_spot_or_futures_historical_klines(self, symbol, kline_size, 
                                              oldest_point, newest_point, market="spot"):
        
        if market == "spot":
            klines = self.binance_client.get_historical_klines(symbol, kline_size, 
                                                               oldest_point, 
                                                               newest_point)
        elif market == "futures":
            klines = self.binance_client.futures_historical_klines(symbol, kline_size, 
                                                               oldest_point, 
                                                               newest_point)
        return klines

    def _get_all_binance(self, symbol, save=False):

        filename = self.data_folder + \
            "/%s-%s-data.csv" % (symbol, self.kline_size)
        if os.path.isfile(filename):
            data_df = pd.read_csv(filename, index_col=0)
        else:
            data_df = pd.DataFrame()
        oldest_point, newest_point = self._minutes_of_new_data(
            symbol, data_df, source="binance")
        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / self._binsizes[self.kline_size])
        if oldest_point == datetime.strptime(self.starting_date, "%d %b %Y"):
            if self.verbose:
                print("Downloading all available %s data for %s. Be patient..!" % (
                    self.kline_size, symbol))
                print(
                    f"starting time: {oldest_point.strftime('%d %b %Y %H:%M:%S')}")
                print(
                    f"ending time: {newest_point.strftime('%d %b %Y %H:%M:%S')}")
                print(f"Downloading {delta_min} minutes of data for {symbol}")
        else:
            if self.verbose:
                print(
                    "Downloading %d minutes of new data available for %s, i.e. %d instances of %s data."
                    % (delta_min, symbol, available_data, self.kline_size)
                )

        try:
            klines = self.get_spot_or_futures_historical_klines(symbol, self.kline_size, 
                                                               oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                               newest_point.strftime("%d %b %Y %H:%M:%S"),
                                                                market=self.market)

        except BinanceAPIException as e:
            print(e)
            print(
                "Something went wrong. Error occured at %s. Wait for 1 hour."
                % (datetime.datetime.now().astimezone(datetime.timezone("UTC")))
            )
            time.sleep(3600)
            self.binance_client = Client(
                self._binance_api_key, self._binance_api_secret)
            klines = self.get_spot_or_futures_historical_klines(symbol, self.kline_size, 
                                                               oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                               newest_point.strftime("%d %b %Y %H:%M:%S"),
                                                                market=self.market)

        data = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "close_time",
                "quote_av",
                "Count",
                "tb_base_av",
                "tb_quote_av",
                "ignore",
            ],
        )
        data["Date"] = pd.to_datetime(data["timestamp"], unit="ms")

        data = data.drop(
            columns=["close_time", "quote_av", "tb_base_av", "tb_quote_av", "ignore"])
        if symbol == 2:
            asset_name = "Bitcoin Cash"
        else:
            asset_name = self.asset_details[self.asset_details["Ticker"]
                                            == symbol]["Asset_Name"].values[0]
        list_features = ["Open", "High", "Low", "Close", "Volume", "Count"]
        renamed_cols = {feature_name: feature_name +
                        asset_name for feature_name in list_features}
        data = data.rename(columns=renamed_cols).copy()
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            temp_df = temp_df.rename(columns=renamed_cols).copy()
#            data_df = data_df.append(temp_df) this is outdated. use concat instead
            data_df = pd.concat([data_df, temp_df])
        else:
            data_df = data
            data_df = data_df.rename(columns=renamed_cols).copy()

        data_df = data_df.drop_duplicates(subset="timestamp")
        if save:
            data_df.to_csv(filename)
        if self.verbose:
            print("All caught up..!")
            print(f"size of dataset: {data_df.shape}")

        return data_df

    def load_cryptos(self, tickers, save=True):

        date_0 = -1
        date_1 = 1

        while date_0 != date_1:
            data_df = pd.DataFrame([])
            for i, symbol in enumerate(tickers):
                data_df_temp = self._get_all_binance(
                    symbol, save=save).reset_index(drop=True)
                date_1 = data_df_temp["timestamp"].iloc[-1]
                if i == 0:
                    date_0 = data_df_temp["timestamp"].iloc[-1]
                if len(data_df) == 0:
                    data_df = pd.concat([data_df, data_df_temp], axis=1)
                else:
                    data_df = pd.concat(
                        [data_df.drop(columns=["timestamp", "Date"]), data_df_temp], axis=1)

        cols_basic_feats = [
            col for col in data_df.columns if col != "timestamp" and col != "Date"]
        data_df[cols_basic_feats] = data_df[cols_basic_feats].astype(float)

        timestamp_start = np.int32(time.mktime(datetime.strptime(
            self.starting_date, "%d %b %Y").timetuple())) * 1e3
        data_df = data_df[data_df["timestamp"] >= timestamp_start]
        data_df["timestamp"] = (data_df["timestamp"] / 1000).astype(int)
        data_df = data_df.set_index("timestamp", drop=False).copy()
        data_df["Future Date"] = (
            data_df["timestamp"] + self.dt * self.period).apply(tu.todatetime).values
        data_df["Date"] = (data_df["timestamp"]).apply(tu.todatetime).values

        return data_df
    
    def get_last_price(self, symbol):
        try:
            # Get the last price for the specified cryptocurrency pair
            ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
            last_price = float(ticker["price"])
            return last_price
        except Exception as e:
            print("Error fetching last price:", e)
            return None

    # def get_account_balance(self, asset):
    #     try:
    #         # Get the account balance for the specified asset
    #         account_info = self.binance_client.get_account()
    #         balances = account_info["balances"]
    #         for balance in balances:
    #             if balance["asset"] == asset:
    #                 return float(balance["free"])
    #         return 0.0  # Return 0 if the asset is not found in the account
    #     except Exception as e:
    #         print("Error fetching account balance:", e)
    #         return None

    def get_account_balance(self, assets, market = "spot"):
        try:
            # Get the account balance for the specified asset
            if market == "spot":
                account_info = self.binance_client.get_account()
            elif market == "futures":
                account_info = self.binance_client.futures_account()
                print(account_info)
            balances = account_info["balances"]

            # Create a DataFrame to store the balances of all assets
            balance_data = {"Asset": [], "Free": [], "Locked": []}

            for balance in balances:
                asset_name = balance["asset"]
                if asset_name in assets:
                    free_balance = float(balance["free"])
                    locked_balance = float(balance["locked"])

                    balance_data["Asset"].append(asset_name)
                    balance_data["Free"].append(free_balance)
                    balance_data["Locked"].append(locked_balance)

            balances_df = pd.DataFrame(balance_data)
            return balances_df

        except Exception as e:
            print("Error fetching account balance:", e)
            return None