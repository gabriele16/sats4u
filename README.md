# sats4u

End-to-end Machine-Learning and Deep-Learning framework to accumulate satoshi's using live-algorithmic trading on binance (1 satoshi is 0.00000001 BTC).

## Installation
Clone repository on Google Drive directory, then open one of the notebooks on Google Colab, adjust path, add binance keys and install dependencies and run an example directly on Colab.
Excute the following command
```
! pip install -r requirements.txt
```

## Implementation
The following functionalities are implemented:
* Download live ticker data from Binance with `python-binance`
* Build technical analysis features using `talib`
* Build CNN-LSTM model for prediction of either next Close, Log of Returns, Buy/Short signal using `Tensorflow`
* Backtest using `backtrader`
* Live-trading using `python-binance`

## Note:
Currently the model is underfitting.
Further improvements:
* Implement some of the features discussed in
[Advances in Financial Machine Learning](https://www.amazon.co.jp/Advances-Financial-Machine-Learning-English-ebook/dp/B079KLDW21), written by Marcos Lopez de Prado.
* Implement a Denoising Autoencoder to select best features, see [Jane Street: Supervised Autoencoder MLP
](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp/notebook)

