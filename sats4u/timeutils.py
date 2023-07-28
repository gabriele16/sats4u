import pandas as pd
import numpy as np
import os, datetime
import time
from datetime import datetime, timedelta

# Use periods = -1 to calculate for labels
# Use periods = 1 or larger to calculate for features
def log_return(series, periods=1):
    log_ret = np.log(series).diff(periods=periods)
    if periods < 0:
        log_ret*= -1.
    return log_ret

def get_utc_timestamp():
    return int(time.time())

def get_next_interval_start(current_time, time_frame):
    interval_duration = parse_time_frame(time_frame)
    return ((current_time // interval_duration) + 1) * interval_duration

def parse_time_frame(time_frame):
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}  # Mapping of time frame units to seconds
    duration, unit = int(time_frame[:-1]), time_frame[-1]
    return duration * units[unit]

def todatetime(timestamp):

    return datetime.fromtimestamp(timestamp)

def totimestamp(string):

    return np.int32(time.mktime(datetime.strptime(string, "%d/%m/%Y").timetuple()))


def shift_time_index(date_time_indexes, period = 1):

    delta_time_index = date_time_indexes[1] - date_time_indexes[0]

    return date_time_indexes.shift(period,freq = delta_time_index)

def arr2series(arr,name,time_indexes,top_or_bottom="bottom"):

    if top_or_bottom == "top":
            return pd.Series(arr,index = time_indexes[:len(arr)],name=name)
    elif top_or_bottom == "bottom":
            return pd.Series(arr,index = time_indexes[-len(arr):],name=name)
    else:
        raise ValueError(
                "Can only merge numpy array with time indexes only if the numpy array starts from the top or bottom"
                )

def mergetimeseries(series1, series2):

    return pd.concat([series1,series2], axis=1)
    
def merge_true_preds(candles_true, preds, columns = ["Close"], period = 1):

    shifted_time_indexes = shift_time_index(candles_true.index,period = period)
    series_predicted = arr2series(preds,"Pred Close",shifted_time_indexes,top_or_bottom="bottom")
    df_preds_true = mergetimeseries(candles_true[columns], series_predicted)

    return df_preds_true

    