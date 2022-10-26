# sats4u

End-to-end Machine-Learning and Deep-Learning framework to accumulate satoshi's using live-algorithmic trading on binance (1 satoshi is 0.00000001 BTC).

## Running the Colab Notebook:

* Install the Google Drive for Desktop (Google Drive)[https://www.google.com/drive/download/]
* Clone repository inside your Google Drive directory
* Open `LSTM-CNN-BitCoin.ipynb` with Google Colab and activate the GPU
* In the Colab notebook adjust path to your own Google Drive, add binance keys files 
* Execute the notebook.

## Dependencies

The dependencies are listed inside the `requirements.txt` file and can be installed via:

```
! pip install -r requirements.txt
```

List of dependencies:

```
python-binance==1.0.16
mplfinance==0.12.9b0
tensorflow==2.8.2
pydot==1.3.0
graphviz==0.10.1
scikit-learn==1.0.2
backtrader==1.9.76.123
pyfolio-reloaded==0.9.3
```

## Implementation
The following functionalities are implemented:
* Download live ticker data from Binance with `python-binance`
* Build technical analysis features using `talib`
* Build CNN-LSTM model for prediction of either next Close, Log of Returns, Buy/Short signal using `Tensorflow`
* Backtest using `backtrader`
* Live-trading using `python-binance`

## Note:
When training over about 100 epochs the model is overfitting and is not profitable upon backtesting.

When trainng for less than 100 epochs the model's accuracy is around 53%. Although this is better than a coin-flip it is still not profitable upon backtesting.

Further improvements:
* Implement some of the functions and features discussed in e.g.
[Advances in Financial Machine Learning](https://www.amazon.co.jp/Advances-Financial-Machine-Learning-English-ebook/dp/B079KLDW21), written by Marcos Lopez de Prado.
* Implement a Denoising Autoencoder to construct noise-free features, see example in [Jane Street: Supervised Autoencoder MLP
](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp/notebook)
* Implement statistical arbitrage as e.g. reported in [Machine Learning for Algorithmic Trading](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d), by Stephan Jansen.

