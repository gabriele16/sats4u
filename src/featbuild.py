import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import timeutils as tu
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


class Candles:
    def __init__(self, cryptodf, cryptoname, target="Close", rollwindow=10):

        what_to_predict = ['Close', 'LogReturns', 'UpDown']
        if target not in what_to_predict:
            raise ValueError(
                "Invalid target to predict. Expected one of: %s" % what_to_predict)

        self.target = target
        self.cryptoname = cryptoname
        self.rollwindow = rollwindow
        self.candles = cryptodf[
            [
                "Date",
                "Low" + cryptoname,
                "High" + cryptoname,
                "Open" + cryptoname,
                "Close" + cryptoname,
                "Volume" + cryptoname,
            ]
        ].copy()
        self.candles.set_index("Date", inplace=True)
        self.candles.rename(
            columns={
                "Low" + cryptoname: "Low",
                "High" + cryptoname: "High",
                "Open" + cryptoname: "Open",
                "Close" + cryptoname: "Close",
                "Volume" + cryptoname: "Volume",
            },
            inplace=True,
        )

        self.dtime_index = self.candles.index[1] - self.candles.index[0]

    def _running_moments(self):

        self.mean = self.candles["Close"].rolling(
            window=self.rollwindow).mean()
        self.std_dev = self.candles["Close"].rolling(
            window=self.rollwindow).std()

    def bbands(self):

        self._running_moments()
        self.candles["UpperBB"] = self.mean + (2 * self.std_dev)
        self.candles["LowerBB"] = self.mean - (2 * self.std_dev)

    def vma(self, length = 9):

        src = self.candles["Close"]
        pdmS = pd.Series(len(src)*[np.nan], dtype='float64')
        mdmS = pd.Series(len(src)*[np.nan], dtype='float64')
        pdiS = pd.Series(len(src)*[np.nan], dtype='float64')
        mdiS = pd.Series(len(src)*[np.nan], dtype='float64')
        iS = pd.Series(len(src)*[np.nan], dtype='float64')
        vma = pd.Series(len(src)*[np.nan], dtype='float64',name="vma")

        k = 1.0 / length
        pdm = np.maximum((src - src.shift(1)), 0)
        mdm = np.maximum((src.shift(1) - src), 0)

        pdmS = pdm.ewm(alpha=k, adjust=False).mean()
        mdmS = mdm.ewm(alpha=k, adjust=False).mean()
        s = pdmS + mdmS
        pdi = pdmS / s
        mdi = mdmS / s
        pdiS = pdi.ewm(alpha=k, adjust=False).mean()
        mdiS = mdi.ewm(alpha=k, adjust=False).mean() 
        d = np.abs(pdiS - mdiS)
        s1 = pdiS + mdiS
        iS = (d / s1).ewm(alpha=k, adjust=False).mean() 

        vI = (iS - iS.rolling(length).min()) / (iS.rolling(length).max() - iS.rolling(length).min())  # Calculate VMA variable factor

        vma = pd.Series(index=src.index,name="vma")
        prev_vma = 0.0
        src = src.fillna(method="backfill")
        vI = vI.fillna(method="backfill")
        vma[0] = src[0]
        for i in range(1,len(src)):
            vma[i] = (1 - k * vI[i]) * prev_vma + k * vI[i] * src[i]
            prev_vma = vma[i]
        
        self.candles = pd.concat([self.candles,vma],axis=1)
    
    def ma(self, window = 31):

        ma = pd.Series(self.candles["Close"].rolling(window=window).mean(), 
                       index=self.candles.index,name="ma" )
        self.candles = pd.concat([self.candles,ma],axis=1)  

    def vma_col(self):

        vmaC = np.where(self.candles["vma"] > self.candles["vma"].shift(1), 1,
                        np.where(self.candles["vma"] < self.candles["vma"].shift(1), -1,
                        np.where(self.candles["vma"] == self.candles["vma"].shift(1), 0, np.nan)))              
        vmaC = pd.Series(vmaC,index=self.candles.index,name="vmaC")

        self.candles = pd.concat([self.candles, vmaC],axis=1)

    def ma_up_low(self):

        self.candles["ma_up_low"] = -1.*(self.candles["Close"] < self.candles["ma"]*(1-1e-4)) + \
                                    1.*(self.candles["Close"] > self.candles["ma"]*(1+1e-4))

    def price2volratio(self):

        self.candles["price2volratio"] = (
            self.candles["Close"] - self.candles["Open"]) / self.candles["Volume"]

    def volacc(self):

        self.candles["vol_diff"] = self.candles["Volume"] - \
            self.candles["Volume"].shift(1)

    def logreturns(self, colname="Close"):

        self.candles["LogReturns"] = tu.log_return(
            self.candles[colname], periods=-1)

    def updowns(self, colname="Close"):

        # use periods = -1. for labels, and multiply by -1 because of diff()
        self.candles["UpDown"] =  (-1.*self.candles["Close"].diff(-1)).apply(
            lambda x: np.nan if np.isnan(x) else 1. if x > 0. else 0.)
        
    def switch2lastcol(self, colname="Close"):

        if self.candles.columns[-1] != colname:
            target_column = self.candles.columns.get_loc(colname)
            last_col = self.candles.columns[-1]
            columns_titles = [colname, last_col]
            cand_temp = self.candles[columns_titles]
            self.candles.iloc[:, -1] = cand_temp[colname]
            self.candles.iloc[:, target_column] = cand_temp.iloc[:, -1]
            self.candles.rename(
                columns={columns_titles[0]: columns_titles[-1],
                         columns_titles[-1]: columns_titles[0]}, inplace=True)

    def buildfeatures(self):

        self.bbands()
        self.vma()
        self.ma()
        self.ma_up_low()
        self.vma_col()
        self.price2volratio()
        self.volacc()
        if self.target == "LogReturns":
            self.logreturns()
        elif self.target == "UpDown":
            self.updowns()
        self.candles = self.candles.iloc[self.rollwindow:]
#        self.candles.fillna(method="pad", inplace=True)
        self.switch2lastcol(colname=self.target)

    def ta_plot(self, in_step=-100):

        title = (
            f"{self.cryptoname} Chart ( {str(self.candles.iloc[in_step].name)} -  {str(self.candles.iloc[-1].name)} )'"
        )

        mpf.plot(self.candles.iloc[in_step:], type="candle", volume=True, figratio=(
            24, 12), style="yahoo", title=title)

    def ta_fullplot(self, in_step=-100, last_step = 0):

        if last_step == 0:
            last_step = len(self.candles)
            last_step_vis = -1
        else:
            last_step_vis = last_step

        title = (
            f"{self.cryptoname} Chart ( {str(self.candles.iloc[in_step].name)} - {str(self.candles.iloc[last_step_vis].name)} )'"
        )

        bollinger_bands_plot = mpf.make_addplot(
            self.candles[["UpperBB", "LowerBB"]].iloc[in_step:last_step], linestyle="dotted")
        price_over_volume_plot = mpf.make_addplot(
            self.candles["price2volratio"].iloc[in_step:last_step], panel=1, color="blue")
        volume_diff_plot = mpf.make_addplot(
            self.candles["vol_diff"].iloc[in_step:last_step], panel=2, type="bar", ylabel="Vol.Acc."
        )

        mpf.plot(
            self.candles.iloc[in_step:last_step],
            type="candle",
            volume=True,
            mav=(3, 11),
            figratio=(24, 12),
            style="yahoo",
            addplot=[bollinger_bands_plot,
                     price_over_volume_plot, volume_diff_plot],
            title=title,
        )

    def ta_vma_plot(self, in_step = -100, last_step = 0, ma_window = 0):

        if last_step == 0:
            last_step = len(self.candles)
            last_step_vis = -1
        else:
            last_step_vis = last_step

        title = (
            f"{self.cryptoname} VMA Chart ( {str(self.candles.iloc[in_step].name)} - {str(self.candles.iloc[last_step_vis].name)} )'"
        )

        ma_red_plot = mpf.make_addplot( (((self.candles['Close']+self.candles["Close"])*0.5 \
                                            < self.candles["ma"]*(1-1e-4))*self.candles["High"]).replace(0.0, np.nan).iloc[in_step:last_step],
                                type='scatter',markersize=15,marker='o', color='r')

        ma_green_plot = mpf.make_addplot( (((self.candles['Close']+self.candles["Close"])*0.5 \
                                            > self.candles["ma"]*(1+1e-4))*self.candles["Low"]).replace(0.0, np.nan).iloc[in_step:last_step],
                                type='scatter',markersize=15,marker='o', color='g')

        vma_plot  = mpf.make_addplot((self.candles['vma']).iloc[in_step:last_step], 
                                      ylabel='vma', secondary_y=False, width = 1)
        
        if (ma_window > 0):
            mpf.plot(self.candles.iloc[in_step:last_step], type='candle', 
                    figratio=(24, 12), style='yahoo',
                    volume=True,
                    mav=(ma_window),
                    addplot=[vma_plot, ma_red_plot, ma_green_plot],title = title)
        else:
            mpf.plot(self.candles.iloc[in_step:last_step], type='candle', 
                    figratio=(24, 12), style='yahoo',
                    volume=True,
                    addplot=[vma_plot, ma_red_plot, ma_green_plot],title = title)  

    def ta_vma_plotly(self, in_step=-100, last_step=0):

        if last_step == 0:
            last_step = len(self.candles)
            last_step_vis = -1
        else:
            last_step_vis = last_step

        title = f"{self.cryptoname} VMA Chart ({str(self.candles.iloc[in_step].name)} - {str(self.candles.iloc[last_step_vis].name)})"

        candlestick = go.Candlestick(x=self.candles.index[in_step:last_step],
                                    open=self.candles['Open'].iloc[in_step:last_step],
                                    high=self.candles['High'].iloc[in_step:last_step],
                                    low=self.candles['Low'].iloc[in_step:last_step],
                                    close=self.candles['Close'].iloc[in_step:last_step],
                                    increasing=dict(line=dict(color='green')),
                                    decreasing=dict(line=dict(color='red')))

        ma_red_scatter = go.Scatter(x=self.candles.index[in_step:last_step],
                                    y=self.candles["High"].iloc[in_step:last_step].where(
                                        (self.candles['Close'] + self.candles["Close"]) * 0.5 <
                                        self.candles["ma"] * (1 - 1e-4)).replace(0.0, float('nan')),
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='circle'))

        ma_green_scatter = go.Scatter(x=self.candles.index[in_step:last_step],
                                    y=self.candles["Low"].iloc[in_step:last_step].where(
                                        (self.candles['Close'] + self.candles["Close"]) * 0.5 >
                                        self.candles["ma"] * (1 + 1e-4)).replace(0.0, float('nan')),
                                    mode='markers',
                                    marker=dict(color='green', size=8, symbol='circle'))

        vma_line = go.Scatter(x=self.candles.index[in_step:last_step],
                            y=self.candles['vma'].iloc[in_step:last_step],
                            mode='lines',
                            name='VMA')
        
        # vma_line = go.Scatter(x=self.candles.index[in_step:last_step],
        #                     y=self.candles['vma'].iloc[in_step:last_step].where(
        #                         ~self.candles['vma'].iloc[in_step:last_step].duplicated(keep=False),
        #                         np.nan),
        #                     mode='lines',
        #                     name='VMA',
        #                     line=dict(color='purple'))

        vma_line_gold = go.Scatter(x=self.candles.index[in_step:last_step],
                                y=self.candles['vma'].iloc[in_step:last_step].where(
                                    self.candles['vma'].iloc[in_step:last_step].duplicated(keep=False),
                                    np.nan),
                                mode='lines',
                                name='Track line',
                                line=dict(color='gold', width=3))

        add_plots = [candlestick, ma_red_scatter, ma_green_scatter, vma_line, vma_line_gold]

        layout = go.Layout(title=title, yaxis=dict(domain=[0.15, 1]), height=600, width=1000)
        fig = go.Figure(data=add_plots, layout=layout)

        return fig