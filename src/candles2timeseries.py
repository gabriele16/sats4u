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

    def denorm(self,values):

        example = self.candles.values[-len(values):,:].copy()
        example[:,-1] = values.squeeze().copy()
        scaled_val = [self.scaler.inverse_transform(np.array([to_scale]))[0][-1] for to_scale in example ]
        return scaled_val
    
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


    def backtest(self, preds, true_vals, split_point, step_back, fee=0.025):


        wallet = 0
        total_wallet_history = []
        single_wallet_history = []

        buys_cnt = 0
        buys_cnt_win = 0
        buys_cnt_losses = 0
        drawback = 0
        old_profit_negative = False
        old_profits = 0

        for i in range(split_point, len(true_vals) - step_back):
            predicted_close = preds[i - split_point]
            previous_close = true_vals[i]
            real_next_close = true_vals[i+1]

            if (previous_close + (previous_close * fee)) < predicted_close:  # buy
                profit = real_next_close - previous_close
                if profit > 0:
                    profit = profit - (profit * fee)
                    buys_cnt_win += 1
                    old_profit_negative = False
                else:
                    profit = profit + (profit * fee)
                    buys_cnt_losses += 1
                    if old_profit_negative:
                        old_profits += profit
                    else:
                        old_profits = profit
                    if old_profits < drawback:
                        drawback = old_profits
                    old_profit_negative = True
                wallet += profit
                total_wallet_history.append(wallet)
                single_wallet_history.append(profit)
                buys_cnt += 1
            else:
                old_profit_negative = False
                old_profits = 0

        print('Fee:', fee)
        print('----------------------')
        print('Buy     ', buys_cnt, '(', buys_cnt_win, 'ok', buys_cnt_losses, 'ko )')
        print('No-op   ', (len(self.x_candles) - split_point) - buys_cnt)
        print('Wallet  ', wallet)
        print('Drawback', drawback)

        return total_wallet_history, single_wallet_history, wallet


