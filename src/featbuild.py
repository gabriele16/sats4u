import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler

class Candles():

    def __init__(self,cryptodf,cryptoname,rollwindow=10):

        self.cryptoname = cryptoname
        self.rollwindow = rollwindow
        self.candles = cryptodf[["Date","Low"+cryptoname,"High"+cryptoname,"Open"+cryptoname,"Close"+cryptoname,"Volume"+cryptoname]].copy()
        self.candles.set_index("Date",inplace = True)
        self.candles.rename(columns = {"Low"+cryptoname:"Low","High"+cryptoname:"High","Open"+cryptoname:"Open",
                    "Close"+cryptoname:"Close","Volume"+cryptoname:"Volume"},inplace=True)
    
    def _running_moments(self):

        self.mean = self.candles["Close"].rolling(window=self.rollwindow).mean()
        self.std_dev = self.candles["Close"].rolling(window=self.rollwindow).std()

    def bbands(self):

        self._running_moments()
        self.candles["UpperBB"] = self.mean + (2*self.std_dev)
        self.candles["LowerBB"] = self.mean - (2*self.std_dev)

    def price2volratio(self):

        self.candles["price2volratio"] = (self.candles['Close'] - self.candles['Open']) / self.candles['Volume']
    
    def volacc(self):

        self.candles['vol_diff'] = self.candles['Volume'] - self.candles['Volume'].shift(1)

    
    def switch2lastcol(self,colname="Close"):

        close_column = self.candles.columns.get_loc(colname)
        last_col = self.candles.columns[-1]
        columns_titles = [colname,last_col]
        cand_temp = self.candles[columns_titles]
        self.candles.iloc[:,-1] = cand_temp[colname]
        self.candles.iloc[:,close_column] = cand_temp.iloc[:,-1]
        self.candles.rename(columns={columns_titles[0]:columns_titles[-1],columns_titles[-1]:columns_titles[0]}, inplace = True)

    def buildfeatures(self):

        self.bbands()
        self.price2volratio()
        self.volacc()
        self.candles=self.candles.iloc[self.rollwindow:]
        self.candles.fillna(method="pad",inplace=True)
        self.switch2lastcol()

    def ta_plot(self,in_step=-100):

        title = f"{self.cryptoname} Chart ( {str(self.candles.iloc[in_step].name)} -  {str(self.candles.iloc[-1].name)} )'"

        mpf.plot(
            self.candles.iloc[in_step:], 
            type='candle', 
            volume=True, 
            figratio=(24,12), 
            style='yahoo', 
            title=title
        )

    def ta_fullplot(self,in_step=-100):

        title = f"{self.cryptoname} Chart ( {str(self.candles.iloc[in_step].name)} - {str(self.candles.iloc[-1].name)} )'"

        bollinger_bands_plot = mpf.make_addplot(self.candles[["UpperBB","LowerBB"]].iloc[in_step:], linestyle='dotted')
        price_over_volume_plot = mpf.make_addplot(self.candles["price2volratio"].iloc[in_step:], panel=1, color='blue')
        volume_diff_plot = mpf.make_addplot(self.candles["vol_diff"].iloc[in_step:], panel=2, type='bar', ylabel='Vol.Acc.')

        mpf.plot(
            self.candles.iloc[in_step:],  
            type='candle', 
            volume=True, 
            mav=(3, 11),
            figratio=(24,12), 
            style='yahoo', 
            addplot=[
                bollinger_bands_plot, 
                price_over_volume_plot, 
                volume_diff_plot
            ], 
            title=title
        )






    








