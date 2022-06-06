import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import time
from sklearn.preprocessing import MinMaxScaler

class Candle2TimeSeries():

    def __init__(self, candles, laststeps = 50000, step_back = 48, candle_step_str = "15m",
                lownorm = 0.2, upnorm= 0.8):

        self.candles = candles
        self.laststeps= laststeps
        self.step_back = step_back
        self.lownorm = lownorm
        self.upnorm = upnorm
        self.candle_step_str = candle_step_str    
        self.scaler = []
        self.candles_norm = []
        self.x_candles = []
        self.x_time = []
        self.y = []

    def normedcandles(self):

        self.scaler = MinMaxScaler(feature_range=(self.lownorm, self.upnorm))
        self.candles_norm = self.scaler.fit_transform(self.candles)

    def denorm(self,value):

        example = [0.5 for x in range(len(self.candles))]
        example[-1] = value
        return self.scaler.inverse_transform([example])[0][-1]
    
    def getlaststeps(self):

        self.candles_norm = self.candles_norm[-self.laststeps:].copy()

    def gettimeseries(self):

        for i in range(len(self.candles_norm) - self.step_back):
            example_candles = []
            example_time = []
            if self.candle_step_str == "1h" or self.candle_step_str == "15m" :
                for o in range(0, self.step_back):
                    example_candles.append(self.candles_norm[i + o])
                    t = self.candles.iloc[ i + o].name
                    example_time.append([t.hour / 24, t.weekday() / 7])
            elif self.candle_step_str == "1m" :
                 for o in range(0, self.step_back):
                    example_candles.append(self.andles_norm[i + o])
                    t = self.candles.iloc[ i + o].name
                    example_time.append([t.minute / 60., t.hour/24])  

            self.x_candles.append(example_candles)
            self.x_time.append(example_time)
            self.y.append(self.candles_norm[i+self.step_back][-1])

    def candles2ts(self):

        self.normedcandles()
        print("Candles Normalized")
        self.getlaststeps()
        print(f"Extracted last {self.laststeps} steps")
        self.gettimeseries()
        print("Generated time-series")
        print(f"Normalized 'candles_norm' with shape : {self.candles_norm.shape}")
        print(f"Feature data 'x_candles' with size : {len(self.x_candles)}")
        print(f"Feature data with time intervals 'x_time' with size : {len(self.x_time)}")


