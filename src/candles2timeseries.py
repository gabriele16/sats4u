import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler


def denorm(scaler, candles, values):

    example = candles.values[-len(values):, :].copy()
    example[:, -1] = values.squeeze().copy()
    scaled_val = [scaler.inverse_transform(np.array([to_scale]))[
        0][-1] for to_scale in example]
    return scaled_val


class Candle2TimeSeries:
    def __init__(self, candles, target="Close", laststeps=50000,
                 step_back=48, candle_step_str="15m", lownorm=0.2, upnorm=0.8):

        what_to_predict = ['Close', 'LogReturns', 'UpDown']

        if target not in what_to_predict:
            raise ValueError(
                "Invalid target to predict, Expected one of: %s" % what_to_predict)

        if target not in candles.columns:
            raise ValueError(f"{target} is not in the candles dataframe")

        self.target = target
        self.candles = candles
        self.laststeps = laststeps
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
        if self.target == "UpDown":
            self.candles_norm[:, -1] = self.candles.iloc[:, -1].values.copy()

    def denorm(self, values):

        example = self.candles.values[-len(values):, :].copy()
        example[:, -1] = values.squeeze().copy()
        scaled_val = [self.scaler.inverse_transform(np.array([to_scale]))[
            0][-1] for to_scale in example]
        return scaled_val

    def getlaststeps(self, timeseries, laststeps):

        if isinstance(timeseries, np.ndarray):
            return timeseries[-laststeps:]
        elif isinstance(timeseries, pd.DataFrame):
            return timeseries.iloc[-laststeps:]
        else:
            ValueError("data_type should be np.array or pd.DataFrame")

    def gettimeseries(self):

        for i in range(len(self.candles_norm) - self.step_back):
            example_candles = []
            example_time = []
            if self.candle_step_str == "1h" or self.candle_step_str == "15m":
                for o in range(0, self.step_back):
                    example_candles.append(self.candles_norm[i + o])
                    t = self.candles.iloc[i + o].name
                    example_time.append([t.hour / 24, t.weekday() / 7])
            elif self.candle_step_str == "1m":
                for o in range(0, self.step_back):
                    example_candles.append(self.andles_norm[i + o])
                    t = self.candles.iloc[i + o].name
                    example_time.append([t.minute / 60.0, t.hour / 24])

            self.x_candles.append(example_candles)
            self.x_time.append(example_time)
            self.y.append(self.candles_norm[i + self.step_back][-1])

    def candles2ts(self, verbose=True):

        self.normedcandles()
        if verbose:
            print("Candles Normalized")
        self.candles_norm = self.getlaststeps(
            self.candles_norm, self.laststeps)
        if verbose:
            print(f"Extracted last {self.laststeps} steps")
        self.gettimeseries()
        if verbose:
            print("Generated time-series")
            print(
                f"Normalized 'candles_norm' with shape : {self.candles_norm.shape}")
            print(
                f"Feature data 'x_candles' with size : {len(self.x_candles)}")
            print(
                f"Feature data with time intervals 'x_time' with size : {len(self.x_time)}")
