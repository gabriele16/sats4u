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
    def __init__(self, cryptoname, target="Close", rollwindow=10):

        what_to_predict = ['Close', 'LogReturns', 'UpDown']
        if target not in what_to_predict:
            raise ValueError(
                "Invalid target to predict. Expected one of: %s" % what_to_predict)

        self.target = target
        self.cryptoname = cryptoname
        self.rollwindow = rollwindow

    def set_candles(self, cryptodf):

        self.candles = cryptodf[
            [
                "Date",
                "Low" + self.cryptoname,
                "High" + self.cryptoname,
                "Open" + self.cryptoname,
                "Close" + self.cryptoname,
                "Volume" + self.cryptoname,
            ]
        ].copy()
        self.candles.set_index("Date", inplace=True)
        self.candles.rename(
            columns={
                "Low" + self.cryptoname: "Low",
                "High" + self.cryptoname: "High",
                "Open" + self.cryptoname: "Open",
                "Close" + self.cryptoname: "Close",
                "Volume" + self.cryptoname: "Volume",
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

    def moving_average(self, x, n, type='simple'):
            """
            compute an n period moving average.

            type is 'simple' | 'exponential'

            """
            x = np.asarray(x)
            if type == 'simple':
                weights = np.ones(n)
            else:
                weights = np.exp(np.linspace(-1., 0., n))

            weights /= weights.sum()

            moving_avg = np.convolve(x, weights, mode='full')[:len(x)]
            moving_avg[:n] = moving_avg[n]
            return moving_avg

    def moving_average_convergence(self, x, nslow=26, nfast=12):
        """
        compute the MACD (Moving Average Convergence/Divergence) using a fast and
        slow exponential moving avg

        return value is emaslow, emafast, macd which are len(x) arrays
        """
        emaslow = self.moving_average(x, nslow, type='exponential')
        emafast = self.moving_average(x, nfast, type='exponential')
        return emaslow, emafast, emafast - emaslow

    def weighted_moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.append(np.array([1]*n), ret[n - 1:] / n)[1:]

    def calcHullMA_inference(self, series, N=50):
        SMA1 = self.weighted_moving_average(series, N)
        SMA2 = self.weighted_moving_average(series, int(N/2))
        res = (2 * SMA2 - SMA1)
        return np.mean(res[-int(np.sqrt(N)):])

        #row["hull"] = last_close - calcHullMA_inference(f[asset]["all_close"][-260:], 240)

    def WMA(self, s, period):
        return s.rolling(period).apply(lambda x: ((np.arange(period)+1)*x).sum()/(np.arange(period)+1).sum(), raw=True)

    def HMA(self, s, period):
        return self.WMA(self.WMA(s, period//2).multiply(2).sub(self.WMA(s, period)), int(np.sqrt(period)))   

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

        self.candles["High"][self.candles["High"] > 80000] = self.candles["Close"][self.candles["High"] > 80000]
        self.bbands()
        self.vma(length = 9)
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

    def upper_shadow(self, df): return df['High'] - np.maximum(df['Close'], df['Open'])
    def lower_shadow(self, df): return np.minimum(df['Close'], df['Open']) - df['Low']

    # Credit: https://www.kaggle.com/swaralipibose/64-new-features-with-autoencoders/code
    def get_features_xgb(self, df_feat, row = False):

        upper_Shadow = self.upper_shadow(df_feat)
        lower_Shadow = self.lower_shadow(df_feat)
        
        df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]

        df_feat['high_div_low_change1'] = df_feat['high_div_low'].pct_change(1)

        df_feat["mom_change1"] = df_feat["Close"].pct_change(1) 

        df_feat["mom_lag3"] = df_feat["Close"].pct_change(3)
        df_feat["mom_lag4"] = df_feat["Close"].pct_change(4)

        df_feat["mom_lag5"] = df_feat["Close"].pct_change(5)
        df_feat["mom_lag7"] = df_feat["Close"].pct_change(7)

        df_feat["rsi_14"] = self.relative_strength_index(df_feat["Close"], n =14)
        df_feat["rsi_7"] = self.relative_strength_index(df_feat["Close"], n =7)
        df_feat["rsi_5"] = self.relative_strength_index(df_feat["Close"], n =5)
        df_feat["rsi_3"] = self.relative_strength_index(df_feat["Close"], n =3)

        df_feat["rsi_3_change3"] = df_feat["rsi_3"].pct_change(3)
        df_feat["rsi_3_change5"] = df_feat["rsi_3"].pct_change(5)

        df_feat["rsi_14_change1"] = df_feat["rsi_14"].pct_change(1)
        df_feat["rsi_14_change5"] = df_feat["rsi_14"].pct_change(5)
        df_feat["rsi_14_change3"] = df_feat["rsi_14"].pct_change(3)

        ma = self.moving_average(df_feat["Close"], n=31)

        ma_slow, ma_fast, df_feat["macd_26_12"] = self.moving_average_convergence(df_feat["Close"], nslow=26, nfast=12)
        ma_slow, ma_fast, df_feat["macd_14_7"] = self.moving_average_convergence(df_feat["Close"], nslow=14, nfast=7)

        df_feat['shadow3'] = upper_Shadow / df_feat['Volume']
        df_feat['shadow5'] = lower_Shadow / df_feat['Volume']

        df_feat['shadow5_chg1'] = df_feat['shadow5'].diff(1)
        df_feat['shadow3_chg1'] = df_feat["shadow3"].diff(1)

        return df_feat

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
            
    def get_vma_dataframe(self, in_step, last_step):

        vma_df = pd.DataFrame({
            "Date": self.candles[in_step:last_step].index,
            "vma_line": self.candles['vma'][in_step:last_step],
            "vma_gold": self.candles['vma'][in_step:last_step].where(
                        self.candles['vma'][in_step:last_step].duplicated(keep=False),
                        np.nan),
            "red_scatter": self.candles["High"][in_step:last_step].where(
                (self.candles['Close'][in_step:last_step] + 
                 self.candles["Close"][in_step:last_step]) * 0.5 <
                self.candles["ma"][in_step:last_step] * (1 - 1e-4)).replace(0.0, np.nan),
            "green_scatter": self.candles["Low"][in_step:last_step].where(
                (self.candles['Close'][in_step:last_step] +
                  self.candles["Close"][in_step:last_step]) * 0.5 >
                self.candles["ma"][in_step:last_step] * (1 + 1e-4)).replace(0.0, np.nan)
        })

        # Add the 'Signal' column based on the trading strategy
        vma_df['Signal'] = 0  # Initialize with 0 (Hold)

        vma_df['Signal'] = np.where(
            (vma_df['green_scatter'].shift(1) > vma_df['vma_line'].shift(1)) &
            (vma_df['green_scatter'].shift(2) > vma_df['vma_line'].shift(2)) &
            (vma_df['green_scatter'] > vma_df['vma_line']),
            1,  # Long signal (Buy)
            vma_df['Signal']
        )

        vma_df['Signal'] = np.where(
            (vma_df['red_scatter'].shift(1) < vma_df['vma_line'].shift(1)) &
            (vma_df['red_scatter'].shift(2) < vma_df['vma_line'].shift(2)) &
            (vma_df['red_scatter'] < vma_df['vma_line']),
            -1,  # Short signal (Sell), if 0 then it is Hold
            vma_df['Signal']
        )

        vma_df["Signal"] = np.where(
            (vma_df["vma_line"] < self.candles['High'][in_step:last_step]) &
            (vma_df["vma_line"] > self.candles['Low'][in_step:last_step] ),
            0, vma_df["Signal"]
        )

        vma_df["Signal"] = np.where(
            pd.notna(vma_df["vma_gold"]), 0, vma_df["Signal"]
        )

        vma_df["Signal"] = np.where(
            (self.candles["RSI"][in_step:last_step] > 80. ) & 
            (vma_df["Signal"] == 1)
            , 0, vma_df["Signal"]
        ) # Do not go long if RSI > 70

        vma_df["Signal"] = np.where(
            (self.candles["RSI"][in_step:last_step] < 20. ) & 
            (vma_df["Signal"] == -1)
            , 0, vma_df["Signal"]
        ) # Do not go short if RSI < 30

        # Calculate the returns in percentage based on the trading signals
        vma_df['Returns'] = vma_df['Signal'].shift(1) * 0.5*(self.candles['Close'][in_step:last_step] +
                                                         self.candles['Open'][in_step:last_step]).pct_change()
        vma_df["Cumulative Returns"] = (1 + vma_df["Returns"]).cumprod() -1.

        return vma_df            

    def ta_vma_plotly(self, in_step, last_step):

        if last_step == 0:
            last_step = len(self.candles)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Dots & Track Line","Signals", "Cumulative Returns"),
                             row_heights=[1.0, 0.3,0.3])

        vma_df = self.get_vma_dataframe(in_step, last_step)

        candlestick = go.Candlestick(x=self.candles[in_step:last_step].index,
                                    open=self.candles['Open'][in_step:last_step],
                                    high=self.candles['High'][in_step:last_step],
                                    low=self.candles['Low'][in_step:last_step],
                                    close=self.candles['Close'][in_step:last_step],
                                    increasing=dict(line=dict(color='green')),
                                    decreasing=dict(line=dict(color='red')))
        fig.add_trace(candlestick, row=1, col=1)

        ma_red_scatter = go.Scatter(x=vma_df.index,
                                    y=vma_df["red_scatter"],
                                    mode='markers',
                                    marker=dict(color='red', size=7, symbol='circle'))
        fig.add_trace(ma_red_scatter, row=1, col=1)

        ma_green_scatter = go.Scatter(x=vma_df.index,
                                    y=vma_df["green_scatter"],
                                    mode='markers',
                                    marker=dict(color='green', size=7, symbol='circle'))
        fig.add_trace(ma_green_scatter, row=1, col=1)

        vma_line = go.Scatter(x=self.candles[in_step:last_step].index,
                            y=self.candles['vma'][in_step:last_step],
                            mode='lines')
        fig.add_trace(vma_line, row=1, col=1)
        
        vma_line_gold = go.Scatter(x=self.candles[in_step:last_step].index,
                                y=self.candles['vma'][in_step:last_step].where(
                                    self.candles['vma'][in_step:last_step].duplicated(keep=False),
                                    np.nan),
                                mode='lines',
                                line=dict(color='gold', width=3))
        fig.add_trace(vma_line_gold, row=1, col=1)

        # Signal
        signal_bar = go.Bar(
            x=vma_df.index,
            y=vma_df["Signal"],
            name="signal"
        )
        fig.add_trace(signal_bar, row=2, col=1)

        # cumulative returns
        cum_returns = go.Scatter(
            x=vma_df.index,
            y=( vma_df["Cumulative Returns"] ),
            mode="lines"
        )
        fig.add_trace(cum_returns, row=3, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            width=1000,
            height=700,
            showlegend=False,
            xaxis_range=[vma_df.index[0], vma_df.index[-1]]
        )

        return fig  

    def ta_fullplot_plotly(self, in_step, last_step):
        if last_step == 0:
            last_step = len(self.candles)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                             subplot_titles=("Bollinger Bands","Relative BBands","RSI", "Volume"),
                             row_heights=[1., 0.3,0.3,0.2]
                             )

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

        avg = ((self.candles["Close"] + self.candles["Open"]).rolling(
            window=self.rollwindow).mean())/2
        # Relative BBands trace
        relative_bbands = ((self.candles["UpperBB"] - self.candles["LowerBB"])/avg)        
        # second_deriv = ((relative_bbands  - 2*relative_bbands.shift(1) +
        #         relative_bbands.shift(2))/relative_bbands)
        # first_deriv = (relative_bbands  - relative_bbands.shift(1))/relative_bbands
        
        relative_avg_trace = go.Scatter(
            x=self.candles[in_step:last_step].index,
            y= relative_bbands[in_step:last_step],
            mode="lines"
        )
        fig.add_trace(relative_avg_trace, row=2, col=1)

        # RSI trace
        rsi_trace = go.Scatter(
            x=self.candles[in_step:last_step].index,
            y=self.candles["RSI"][in_step:last_step],
            mode="lines"
        )
        fig.add_trace(rsi_trace, row=3, col=1)

        # Volume trace
        volume_trace = go.Bar(
            x=self.candles[in_step:last_step].index,
            y=self.candles["Volume"][in_step:last_step],
            name="Volume"
        )
        fig.add_trace(volume_trace, row=4, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            width=1000,
            height=700,
            showlegend=False,
        )

        return fig
