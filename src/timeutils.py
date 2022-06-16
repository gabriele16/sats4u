import pandas as pd
import numpy as np
import os, datetime
import time
from datetime import datetime, timedelta


def log_return(series, periods=1):
    log_ret = np.log(series).diff(periods=periods)
    if periods < 0:
        log_ret*= -1.
    return log_ret

def todatetime(timestamp):

    return datetime.fromtimestamp(timestamp)

def totimestamp(string):

    return np.int32(time.mktime(datetime.strptime(string, "%d/%m/%Y").timetuple()))


def shift_time_index(date_time_indexes,period,delta_time_index):

    return date_time_indexes.shift(period,freq = delta_time_index)

def arr2series(arr,name,time_indexes,top_or_bottom):

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

    