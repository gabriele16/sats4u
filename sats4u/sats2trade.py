import os, sys
import time
import pandas as pd
from . import loadcrypto as lc
from . import featbuild as fb
from . import timeutils as tu

# Define the Sats2Trade class that inherits from CryptoData and Candles
class Sats2Trade(lc.CryptoData, fb.Candles):
    def __init__(self, crypto_pair):
        # Call the __init__ methods of the parent classes explicitly
        self.set_data_folder_and_asset_details()
        lc.CryptoData.__init__(self, self.asset_details, self.data_folder)

        self.crypto_pair = crypto_pair
        cryptoname = self.crypto_pair_dict[crypto_pair]
        fb.Candles.__init__(self, cryptoname, target="UpDown", rollwindow=10)

        # Initialize an empty dataframe to store the returns and cumulative returns
        self.returns_df = pd.DataFrame(columns=["Date", "Returns", "Cumulative Returns"])

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
            print(f"Buy order {order['orderId']} placed successfully: {order}")
        except Exception as e:
            print("Error placing buy order:", e)

    # Function to place a sell order
    def place_sell_order(self, quantity):
        try:
            order = self.binance_client.create_order(
                symbol=self.crypto_pair,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            print(f"Sell order {order['orderId']} placed successfully: {order}")
        except Exception as e:
            print("Error placing sell order:", e)

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
            print(f"Short sell order {order['orderId']} placed successfully: {order}")
        except Exception as e:
            print("Error placing short sell order:", e)

    # Function to update the technical indicators and signals
    def update_trade_signal(self, in_step = 0, last_step = -1):
        # Your code to update the technical indicators here

        self.buildfeatures()
        # Get the signals DataFrame
        signals_df = self.get_vma_dataframe(in_step, last_step)
        return signals_df        

    def get_portfolio(self, balances):
        # Calculate the total portfolio value in USDT
        total_value = 0
        for asset in balances:
            if asset['asset'] == 'USDT':
                total_value += float(asset['free']) + float(asset['locked'])
            else:
                asset_symbol = asset['asset'] + 'USDT'
                asset_price = self.get_last_price(asset_symbol)
                total_value += (float(asset['free']) + float(asset['locked'])) * asset_price

        # Calculate the current returns and cumulative returns
        current_returns = (total_value / self.initial_balance) - 1
        cumulative_returns = current_returns + 1

        # If portfolio_df is not defined, create it
        if not hasattr(self, 'portfolio_df'):
            self.portfolio_df = pd.DataFrame(columns=["Date", "Total Balance", "Returns", "Cumulative Returns"])

        # Append the new data to the portfolio_df
        self.portfolio_df = self.portfolio_df.append(
            {
                "Date": pd.Timestamp.now(),
                "Total Balance": total_value,
                "Returns": current_returns,
                "Cumulative Returns": cumulative_returns,
            },
            ignore_index=True,
        )

        return total_value, current_returns, cumulative_returns

    def trade_loop(self, quantity):
        previous_signal = 0
        current_position = 0
        self.initial_balance = self.get_account_balance("USDT")  # Store the initial balance for calculating returns

        while True:
            try:
                # Get new candles data
                ldatadf = self.load_cryptos(self.crypto_pair, save=True)
                self.set_candles(ldatadf)

                # Update trade signal based on technical analysis only for now
                signals_df = self.update_trade_signal(self.candles)

                # Get the last signal (latest entry in the DataFrame)
                last_signal = signals_df.iloc[-1]['Signal']

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
                balances = self.get_account_balance()
                total_balance, current_returns, cumulative_returns = self.get_portfolio(balances)

                # Set the previous signal for the next iteration
                previous_signal = last_signal
                # Update the current position based on the signal
                current_position = last_signal

                # Sleep for a certain period before the next iteration
                time.sleep(60)  # Sleep for 60 seconds (adjust this based on your strategy)
            except Exception as e:
                print("Error occurred:", e)


# def strategy(pair, entry, lookback, qty, open_position=False):
#     while True:
#         df = pd.read_sql(pair, engine)
#         df.head()
#         lookbackperiod = df.iloc[-lookback:]
#         cumret = (lookbackperiod.Price.pct_change() +1).cumprod() - 1
#         if not open_position:
#             print(open_position)
#             if cumret[cumret.last_valid_index()] > entry:
#                 order = client.create_order(symbol=pair,
#                                            side='BUY',
#                                            type='MARKET',
#                                            quantity=qty)
#                 print(order)
#                 open_position = True
#                 break
#     if open_position:
#         while True:
#             df = pd.read_sql(pair, engine)
#             sincebuy = df.loc[df.Time > 
#                               pd.to_datetime(order['transactTime'],
#                                             unit='ms')]
#             if len(sincebuy) > 1:
#                 sincebuyret = (sincebuy.Price.pct_change() +1).cumprod() - 1
#                 last_entry = sincebuyret[sincebuyret.last_valid_index()]
#                 if last_entry > 0.0015 or last_entry < -0.0015:
#                     order = client.create_order(symbol=pair,
#                                            side='SELL',
#                                            type='MARKET',
#                                            quantity=qty)
#                     print(order)
#                     break
 
#def tradesats(investment=100):