import numpy as np
import pandas as pd
import math
import os, datetime
import time
from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

from binance import BinanceSocketManager
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException # here
from binance.client import Client
from dateutil import parser

from pathlib import Path
#from warnings import simplefilter

from tqdm import tqdm

import traceback
import json


class CryptoData():

    def __init__(self,asset_details,data_folder):

        self.asset_details = asset_details
        self.data_folder = data_folder
        
        self.tick_to_id = {"BTCUSDT":1,"BCHUSDT":2,"BNBUSDT":0,"EOSUSDT":5,"ETCUSDT":7,"ETHUSDT":6,
                    "LTCUSDT":9,"XMRUSDT":11,"TRXUSDT":13,"XLMUSDT":12,"ADAUSDT":3,
                    "IOTAUSDT":8,"MKRUSDT":10, "DOGEUSDT":4}
        
        self.tick_to_id_df = pd.DataFrame(np.array([list(self.tick_to_id.keys()) ,list(self.tick_to_id.values())]).T,columns=["Ticker","Asset_ID"])
        self.tick_to_id_df["Asset_ID"]=self.tick_to_id_df["Asset_ID"].astype(int)
        self.asset_details = pd.merge(self.asset_details,self.tick_to_id_df,on="Asset_ID")

    def _binance_api_constants(self):

        self._binsizes = {"1m": 1, "5m": 5, "15m":15 ,"1h": 60, "1d": 1440}
        self._batch_size = 750

    def trade_time_units(self,dt=60,kline_size="1m",starting_date = '1 Mar 2022'):

        self.dt = dt
        self.kline_size = kline_size
        self.starting_date = starting_date
        self.dt_daily = 60*60*24
        self.minute = 1*self.dt
        self.hour = 60*self.minute
        self.day = self.hour*24
        self.week = self.day*7
        self.month = self.day*30
        self.year = self.day*365
        self._binance_api_constants()

    def load_binance_client(self,secrets_filename,data1_str = 'DATA1',data2_str = 'DATA2i'):

        #secrets_filename = data_folder+'crypto_data/data.json'
        api_keys = {}
        with open(secrets_filename, 'r') as f:
            api_keys = json.loads(f.read())

        ### API
        self._binance_api_key = api_keys[data1_str]    #Enter your own API-key here
        self._binance_api_secret = api_keys[data2_str] #Enter your own API-secret here
        self._binance_client = Client(self._binance_api_key,self._binance_api_secret)

### FUNCTIONS
    def _minutes_of_new_data(self,symbol, data, source):

        if len(data) > 0:  
            old = parser.parse(data["Date"].iloc[-1])
        elif source == "binance": 
            old = datetime.strptime(self.starting_date, '%d %b %Y')
        if source == "binance":      
            try:
                new = pd.to_datetime(self._binance_client.get_klines(symbol=symbol, interval=self.kline_size)[-1][0], unit='ms')
            except BinanceAPIException as e:
                print(e)
                print('Something went wrong. Error occured at %s. Wait for 1 hour.' % (datetime.datetime.now().astimezone(timezone('UTC'))))
                time.sleep(3600)
                self._binance_client = Client(self._binance_api_key, self._binance_api_secret)
                new = pd.to_datetime(self._binance_client.get_klines(symbol=symbol, interval=self.kline_size)[-1][0], unit='ms')                
                
        return old, new

    def _get_all_binance(self, symbol, save = False):

        filename = self.data_folder+'/%s-%s-data.csv' % (symbol, self.kline_size)
        if os.path.isfile(filename): 
            data_df = pd.read_csv(filename,index_col = 0)
        else: 
            data_df = pd.DataFrame()
        oldest_point, newest_point = self._minutes_of_new_data(symbol,data_df, source = "binance")
        delta_min = (newest_point - oldest_point).total_seconds()/60
        available_data = math.ceil(delta_min/self._binsizes[self.kline_size])
        if oldest_point == datetime.strptime(self.starting_date, '%d %b %Y'):
            print('Downloading all available %s data for %s. Be patient..!' % (self.kline_size, symbol))
            print(f"starting time: {oldest_point.strftime('%d %b %Y %H:%M:%S')}")
            print(f"ending time: {newest_point.strftime('%d %b %Y %H:%M:%S')}")
            print(f"Downloading {delta_min} minutes of data for {symbol}")
        else: 
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, self.kline_size))

        try:
            klines = self._binance_client.get_historical_klines(symbol, self.kline_size, 
                                                        oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                        newest_point.strftime("%d %b %Y %H:%M:%S"))
        except BinanceAPIException as e:
            print(e)
            print('Something went wrong. Error occured at %s. Wait for 1 hour.' % (datetime.datetime.now().astimezone(timezone('UTC'))))
            time.sleep(3600)
            self._binance_client = Client(self._binance_api_key, self._binance_api_secret)
            klines = self._binance_client.get_historical_klines(symbol, self.kline_size, 
                                                        oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                        newest_point.strftime("%d %b %Y %H:%M:%S"))

        data = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                            'close_time', 'quote_av', 'Count', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')

        data=data.drop(columns = ["close_time","quote_av","tb_base_av","tb_quote_av","ignore"])
        if symbol ==2:
            asset_name = "Bitcoin Cash"
        else:
            asset_name = self.asset_details[self.asset_details["Ticker"]==symbol]["Asset_Name"].values[0]
        list_features = ['Open', 'High', 'Low', 'Close', 'Volume',"Count"]
        renamed_cols = {feature_name: feature_name+asset_name for feature_name in list_features}
        data = data.rename(columns=renamed_cols).copy()
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            temp_df = temp_df.rename(columns=renamed_cols).copy()
            data_df = data_df.append(temp_df)
        else: 
            data_df = data
            data_df = data_df.rename(columns=renamed_cols).copy()

        data_df = data_df.drop_duplicates(subset="timestamp")
        if save: 
            data_df.to_csv(filename)
        print('All caught up..!')
        print(f'size of dataset: {data_df.shape}')

        return data_df

    def load_cryptos(self, tickers, save = True):

        data_size_0 = -1
        data_size = 1
                
        while data_size_0 != data_size:
            data_df=pd.DataFrame([])
            for i,symbol in enumerate(tickers):
                data_df_temp = self._get_all_binance(symbol, save = save).reset_index(drop=True)
                data_size = data_df_temp.shape[0]
                if i == 0:
                    data_size_0 = data_df_temp.shape[0]
                if len(data_df) ==0:
                    data_df = pd.concat([data_df,data_df_temp],axis=1)
                else:
                    data_df = pd.concat([data_df.drop(columns=["timestamp","Date"])
                                        ,data_df_temp],axis=1)
        
        list_basic_features = ['Open', 'High', 'Low', 'Close', 'Volume',"Count"]
        cols_basic_feats = [col for col in data_df.columns if col !="timestamp" and col != "Date"]
        data_df[cols_basic_feats] = data_df[cols_basic_feats].astype(float)
        
        timestamp_start  = np.int32(time.mktime(datetime.strptime(self.starting_date, '%d %b %Y').timetuple()))*1e3
        data_df = data_df[data_df["timestamp"] >=  timestamp_start]
        return data_df