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
* Implement some of the functions and features discussed in e.g.
[Advances in Financial Machine Learning](https://www.amazon.co.jp/Advances-Financial-Machine-Learning-English-ebook/dp/B079KLDW21), written by Marcos Lopez de Prado.
* Implement a Denoising Autoencoder to select best features, see [Jane Street: Supervised Autoencoder MLP
](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp/notebook)
* Implement statistical arbitrage as e.g. reported in [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d), by Stephan Jansen.

