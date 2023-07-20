import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from sats4u import timeutils as tu
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    def relative_strength_index(self, n=14):
        """
        compute the n period relative strength indicator
        http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
        http://www.investopedia.com/terms/r/rsi.asp
        """

        deltas = np.diff(self.candles["Close"])
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(self.candles["Close"])
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(self.candles["Close"])):
            delta = deltas[i - 1]  # cause the diff is 1 shorter

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        
        self.candles["RSI"] = rsi


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

        vma = pd.Series(index=src.index,name="vma", dtype='float64')
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
        self.relative_strength_index()
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

    def ta_vma_plotly(self, in_step, last_step):

        if last_step == 0:
            last_step = len(self.candles)
            last_step_vis = -1
        else:
            last_step_vis = last_step

        title = f"{self.cryptoname} Dots & Track Line" # ({str(self.candles[in_step].name)} - {str(self.candles[last_step_vis].name)})"

        candlestick = go.Candlestick(x=self.candles[in_step:last_step].index,
                                    open=self.candles['Open'][in_step:last_step],
                                    high=self.candles['High'][in_step:last_step],
                                    low=self.candles['Low'][in_step:last_step],
                                    close=self.candles['Close'][in_step:last_step],
                                    increasing=dict(line=dict(color='green')),
                                    decreasing=dict(line=dict(color='red')))

        ma_red_scatter = go.Scatter(x=self.candles[in_step:last_step].index,
                                    y=self.candles["High"][in_step:last_step].where(
                                        (self.candles['Close'] + self.candles["Close"]) * 0.5 <
                                        self.candles["ma"] * (1 - 1e-4)).replace(0.0, float('nan')),
                                    mode='markers',
                                    marker=dict(color='red', size=7, symbol='circle'))

        ma_green_scatter = go.Scatter(x=self.candles[in_step:last_step].index,
                                    y=(self.candles["Low"][in_step:last_step]).where(
                                        (self.candles['Close'] + self.candles["Close"]) * 0.5 >
                                        self.candles["ma"] * (1 + 1e-4)).replace(0.0, float('nan')),
                                    mode='markers',
                                    marker=dict(color='green', size=7, symbol='circle'))

        vma_line = go.Scatter(x=self.candles[in_step:last_step].index,
                            y=self.candles['vma'][in_step:last_step],
                            mode='lines')
        
        vma_line_gold = go.Scatter(x=self.candles[in_step:last_step].index,
                                y=self.candles['vma'][in_step:last_step].where(
                                    self.candles['vma'][in_step:last_step].duplicated(keep=False),
                                    np.nan),
                                mode='lines',
                                line=dict(color='gold', width=3))

        add_plots = [candlestick, ma_red_scatter, ma_green_scatter, vma_line, vma_line_gold]

        layout = go.Layout(title=title, yaxis=dict(domain=[0.05, 1]), height=700, width=1000)
        fig = go.Figure(data=add_plots, layout=layout)
        fig.update_layout(showlegend=False, xaxis_rangeslider_visible=False)

        return fig

    def ta_fullplot_plotly(self, in_step, last_step):
        if last_step == 0:
            last_step = len(self.candles)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Bollinger Bands","RSI", "Volume"),
                             row_heights=[0.9, 0.3,0.3])

        # Candlestick trace
        candlestick_trace = go.Candlestick(
            x=self.candles[in_step:last_step].index,
            open=self.candles["Open"][in_step:last_step],
            high=self.candles["High"][in_step:last_step],
            low=self.candles["Low"][in_step:last_step],
            close=self.candles["Close"][in_step:last_step],
        )
        fig.add_trace(candlestick_trace, row=1, col=1)

        # Bollinger Bands trace
        bollinger_bands_trace = go.Scatter(
            x=self.candles[in_step:last_step].index,
            y=self.candles["UpperBB"][in_step:last_step],
            mode="lines"
        )
        fig.add_trace(bollinger_bands_trace, row=1, col=1)

        bollinger_bands_trace = go.Scatter(
            x=self.candles[in_step:last_step].index,
            y=self.candles["LowerBB"][in_step:last_step],
            mode="lines"
        )
        fig.add_trace(bollinger_bands_trace, row=1, col=1)

        # RSI trace
        rsi_trace = go.Scatter(
            x=self.candles[in_step:last_step].index,
            y=self.candles["RSI"][in_step:last_step],
            mode="lines"
        )
        fig.add_trace(rsi_trace, row=2, col=1)

        # Volume trace
        volume_trace = go.Bar(
            x=self.candles[in_step:last_step].index,
            y=self.candles["Volume"][in_step:last_step],
            name="Volume"
        )
        fig.add_trace(volume_trace, row=3, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            width=1000,
            height=700,
            showlegend=False,
            yaxis=dict(
                domain=[0.5, 1]  # Adjust the domain to allocate more space for the volume plot
            )
        )

        return fig
