import numpy as np
import pandas as pd
import os, datetime
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
import math

from binance import BinanceSocketManager
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException # here

from binance.client import Client
from datetime import timedelta
from dateutil import parser

from pathlib import Path
from warnings import simplefilter

from tqdm import tqdm

import os
import traceback

import json
from google.colab import drive



#drive.mount('/content/drive')
#data_folder = "/content/drive/MyDrive/Colab Notebooks/crypto_project/"
#asset_details = pd.read_csv(data_folder + 'crypto_data/asset_details.csv')




def assets_with_tickers(asset_details):

    tick_to_id = {"BTCUSDT":1,"BCHUSDT":2,"BNBUSDT":0,"EOSUSDT":5,"ETCUSDT":7,"ETHUSDT":6,
                "LTCUSDT":9,"XMRUSDT":11,"TRXUSDT":13,"XLMUSDT":12,"ADAUSDT":3,
                  "IOTAUSDT":8,"MKRUSDT":10, "DOGEUSDT":4}
     
    tick_to_id_df = pd.DataFrame(np.array([list(tick_to_id.keys()) ,list(tick_to_id.values())]).T,columns=["Ticker","Asset_ID"])
    tick_to_id_df["Asset_ID"]=tick_to_id_df["Asset_ID"].astype(int)
     
    asset_details_ticks = pd.merge(asset_details,tick_to_id_df,on="Asset_ID")

    return asset_details_ticks

def trade_time_units(DT):

     DT = 60
     DT_DAILY = 60*60*24
     
     minute = 1*DT
     hour = 60*minute
     day = hour*24
     week = day*7
     month = day*30
     year = day*365

     return  DT_DAILY, minute, hour, day, week, month, year


def binance_api_constants(starting_date = '1 Mar 2022'):

    binsizes = {"1m": 1, "5m": 5, "15m":15 ,"1h": 60, "1d": 1440}
    batch_size = 750
    STARTING_DATE = starting_date

    return binsizes, batch_size, STARTING_DATE


def load_binance_client(secrets_filename,data1_str = 'DATA1',data2_str = 'DATA2i'):

    #secrets_filename = data_folder+'crypto_data/data.json'
    api_keys = {}
    with open(secrets_filename, 'r') as f:
        api_keys = json.loads(f.read())

    ### API
    binance_api_key = api_keys[data1_str]    #Enter your own API-key here
    binance_api_secret = api_keys[data2_str] #Enter your own API-secret here

    binance_client = Client(binance_api_key,binance_api_secret)
    
    return binance_client, binance_api_key, binance_api_secret

### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, binance_client, binance_api_key, binance_api_secret, 
                        data, source, starting_date = '1 Mar 2022'):

    if len(data) > 0:  
        old = parser.parse(data["Date"].iloc[-1])
    elif source == "binance": 
        old = datetime.strptime(starting_date, '%d %b %Y')
    if source == "binance":      
        try:
            new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
        except BinanceAPIException as e:
            print(e)
            print('Something went wrong. Error occured at %s. Wait for 1 hour.' % (datetime.datetime.now().astimezone(timezone('UTC'))))
            time.sleep(3600)
            binance_client = Client(binance_api_key, binance_api_secret)
            new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')                
            
    return old, new

def get_all_binance(data_folder, symbol, binsizes, kline_size, 
                    binance_client, binance_api_key, binance_api_secret, asset_details_ticks,
                    save = False,starting_date = '1 Mar 2022'):

    filename = data_folder+'/%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): 
        data_df = pd.read_csv(filename,index_col = 0)
    else: 
        data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, 
                                 binance_client, binance_api_key, binance_api_secret, 
                                 data_df, source = "binance",starting_date = starting_date)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime(starting_date, '%d %b %Y'):
        print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
        print(f"starting time: {oldest_point.strftime('%d %b %Y %H:%M:%S')}")
        print(f"ending time: {newest_point.strftime('%d %b %Y %H:%M:%S')}")
        print(f"Downloading {delta_min} minutes of data for {symbol}")
    else: 
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))

    try:
        klines = binance_client.get_historical_klines(symbol, kline_size, 
                                                      oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                      newest_point.strftime("%d %b %Y %H:%M:%S"))
    except BinanceAPIException as e:
        print(e)
        print('Something went wrong. Error occured at %s. Wait for 1 hour.' % (datetime.datetime.now().astimezone(timezone('UTC'))))
        time.sleep(3600)
        binance_client = Client(binance_api_key, binance_api_secret)
        klines = binance_client.get_historical_klines(symbol, kline_size, 
                                                      oldest_point.strftime("%d %b %Y %H:%M:%S"), 
                                                      newest_point.strftime("%d %b %Y %H:%M:%S"))

    data = pd.DataFrame(klines, columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'close_time', 'quote_av', 'Count', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['Date'] = pd.to_datetime(data['timestamp'], unit='ms')

    data=data.drop(columns = ["close_time","quote_av","tb_base_av","tb_quote_av","ignore"])
    if symbol ==2:
        asset_name = "Bitcoin Cash"
    else:
        asset_name = asset_details_ticks[asset_details_ticks["Ticker"]==symbol]["Asset_Name"].values[0]
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



def load_loop_over_cryptos(data_folder, ticker, binsizes, kline_size, 
                           binance_client, binance_api_key, binance_api_secret, 
                           asset_details_ticks, save = True, starting_date='1 Mar 2022'):

    data_size_0 = -1
    data_size = 1
    while data_size_0 != data_size:
        data_df=pd.DataFrame([])
        for i,symbol in enumerate(ticker):
            data_df_temp = get_all_binance(data_folder, symbol, binsizes, kline_size, 
                            binance_client, binance_api_key, binance_api_secret,
                            asset_details_ticks, save = save, starting_date = starting_date).reset_index(drop=True)
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

    return data_df

