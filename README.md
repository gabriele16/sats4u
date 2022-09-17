# sats4u

End-to-end framework to accumulate satoshi's using live-algorithmic trading on binance aided by deep-learning (1 satoshi is 0.00000001 BTC).

# Installation
Clone on your Google Drive directory, then open one of the notebooks on Google Colab, adjust path, add binance keys and install dependencies and run an example directly on Colab.
Excute the following command
```
! pip install -r requirements.txt
```

# Implementation
The following functionalities are implemented:
* Download live ticker data from Binance with `python-binance`
* Build technical analysis features using `talib`
* Build CNN-LSTM model for prediction of either next Close, Log of Returns, Buy/Short signal using `Tensorflow`
* Backtest using `backtrader`
* Live-trading using `python-binance`
