import logging
import os, sys
import time
import pandas as pd
from . import loadcrypto as lc
from . import featbuild as fb
from . import timeutils as tu

# Define the Sats2Trade class that inherits from CryptoData and Candles
class Sats2Trade(lc.CryptoData, fb.Candles):
    def __init__(self, crypto_pair="BTCUSDT", time_frame="15m"):
        # Call the __init__ methods of the parent classes explicitly
        self.time_frame = time_frame
        self.set_data_folder_and_asset_details()
        lc.CryptoData.__init__(self, self.asset_details, self.data_folder)

        self.trade_time_units(dt=60, kline_size=self.time_frame,
                               starting_date="1 Mar 2022")
        
        self.reference_currency = "USDT"

        if self.reference_currency not in crypto_pair:
            raise ValueError(f"The crypto pair must be priced in the reference currency: \
                              {self.reference_currency}")

        self.crypto_pair = crypto_pair
        self.assets_to_trade = [self.reference_currency, 
                                crypto_pair.split(self.reference_currency)]
        cryptoname = self.crypto_pair_dict[crypto_pair]
        fb.Candles.__init__(self, cryptoname, target="UpDown", rollwindow=10)

    def set_data_folder_and_asset_details(self):

        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, "..")
        self.data_folder = os.path.join(root_dir, "data")
        self.asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))

    # Function to place a buy order
    def place_buy_order(self, quantity):
        try:
            order = self.binance_client.create_order(
                symbol=self.crypto_pair,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            logging.info(f"Buy order {order['orderId']} placed successfully: {order} at {tu.get_utc_timestamp()}")
        except Exception as e:
            logging.info("Error placing buy order:", e)

    # Function to place a sell order
    def place_sell_order(self, quantity):
        try:
            order = self.binance_client.create_order(
                symbol=self.crypto_pair,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            logging.info(f"Sell order {order['orderId']} placed successfully: {order} at {tu.get_utc_timestamp()}")
        except Exception as e:
            logging.info("Error placing sell order:", e)

    # Function to place a short sell order
    def place_short_sell_order(self, quantity):
        try:
            order = self.binance_client.create_order(
                symbol=self.crypto_pair,
                side='SELL',
                type='MARKET',
                quantity=quantity,
                isIsolated='TRUE'  # Enable this if using isolated margin for short selling
            )
            logging.info(f"Short sell order {order['orderId']} placed successfully: {order} at {tu.get_utc_timestamp()}")
        except Exception as e:
            logging.info("Error placing short sell order:", e)

    # Function to update the technical indicators and signals
    def update_trade_signal(self):

        self.buildfeatures()
        # Get the signals DataFrame
        in_step = self.rollwindow
        last_step = self.candles.shape[0]
        signals_df = self.get_vma_dataframe(in_step, last_step)
        return signals_df        

    def get_portfolio(self, balances, model_signal = 0, 
                        model_current_returns = 0,
                        model_cumulative_returns = 1):
        
        # Calculate the total portfolio value in USDT
        total_value = 0
        for asset in balances["Asset"].values:
            if asset == self.reference_currency :
                total_value += float(balances[balances["Asset"] == asset]['Free'].values) + \
                float(balances[balances["Asset"] == asset]['Locked'].values)
            else:
                asset_symbol = asset + self.reference_currency
                asset_price = self.get_last_price(asset_symbol)
                if asset_price is not None:
                    total_value += (float(balances[balances["Asset"] == asset]['Free'].values) + \
                                 float(balances[balances["Asset"] == asset]['Locked'].values)) * asset_price

        # Calculate the current returns and cumulative returns
        if not hasattr(self, 'initial_balance'):
            self.initial_balance = total_value

        current_returns = (total_value / self.initial_balance) - 1
        cumulative_returns = current_returns + 1

        # If portfolio_df is not defined, create it
        if not hasattr(self, 'portfolio_df'):
            self.portfolio_df = pd.DataFrame(columns=["Date", "Total Balance",
                                                       "Returns", "Cumulative Returns",
                                                       "Model Returns", "Cumulative Model Returns",
                                                       "Model Signal"])
            self.portfolio_df.to_csv('portfolio.csv', mode='a', header=True, index=False)
                                           

        # Append the new data to the portfolio_df
        self.portfolio_df = self.portfolio_df.append(
            {
                "Date": pd.Timestamp.now(),
                "Total Balance": total_value,
                "Returns": current_returns,
                "Cumulative Returns": cumulative_returns,
                "Model Returns": model_current_returns,
                "Cumulative Model Returns": model_cumulative_returns,
                "Model Signal": model_signal,
            },
            ignore_index=True,
        )
        self.portfolio_df.to_csv('portfolio.csv', mode='a', header=False, index=False)


        return total_value, current_returns, cumulative_returns

    def trade_loop(self, quantity):
        previous_signal = 0
        current_position = 0
        balances = self.get_account_balance(self.assets_to_trade)
        self.initial_balance, current_returns, cumulative_returns = self.get_portfolio(balances)  # Store the initial balance for calculating returns
        # Set up logging
        logging.basicConfig(filename='trade_loop.log', level=logging.INFO,
                             format='%(asctime)s - %(levelname)s - %(message)s')

        while True:
            #try:

                tu.wait_for_next_interval(self.time_frame)
                # Get new candles data
                logging.info("Getting new candles data...")
                ldatadf = self.load_cryptos([self.crypto_pair], save=True)
                self.set_candles(ldatadf)

                # Update trade signal based on technical analysis only for now
                signals_df = self.update_trade_signal()

                # Get the last signal (latest entry in the DataFrame)
                last_signal = signals_df.iloc[-1]['Signal']
                # logging.info("Signal DataFrame:")
                # logging.info(signals_df.tail(1))
                # Append signals_df to signals.csv
                signals_df.to_csv('signals.csv', mode='a', header=True, index=False)
                model_current_returns = signals_df.iloc[-1]['Returns']
                model_cumulative_returns = signals_df.iloc[-1]['Cumulative Returns']

                # If the signal changes from 0 to 1, place a buy order
                if previous_signal == 0 and last_signal == 1:
                    self.place_buy_order(quantity)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance -= order_value
                    current_position = 1

                # If the signal changes from 1 to 0 and there is an open position, close the position
                if previous_signal == 1 and last_signal == 0 and current_position == 1:
                    self.place_sell_order(quantity)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance += order_value
                    current_position = 0

                # If the signal changes from 0 to -1, place a short sell order
                if previous_signal == 0 and last_signal == -1:
                    self.place_short_sell_order(quantity)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance -= order_value
                    current_position = -1

                # If the signal changes from -1 to 0 and there is an open short position, close the short position
                if previous_signal == -1 and last_signal == 0 and current_position == -1:
                    self.place_sell_order(quantity)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance += order_value
                    current_position = 0

                # Calculate the portfolio details
                balances = self.get_account_balance(self.assets_to_trade)
                total_balance, current_returns, cumulative_returns = self.get_portfolio(balances,
                                                                                         model_signal=last_signal,
                                                                                         model_current_returns=model_current_returns,
                                                                                         model_cumulative_returns=model_cumulative_returns)
                logging.info("Portfolio DataFrame:")
                logging.info(self.portfolio_df.tail(1))

                # Set the previous signal for the next iteration
                previous_signal = last_signal
                # Update the current position based on the signal
                current_position = last_signal

                # Sleep for a certain period before the next iteration
                time.sleep(60)  # Sleep for 60 seconds (adjust this based on your strategy)
            #except Exception as e:
            #    print("Error occurred:", e)
