import logging
import os, sys
import time
import pandas as pd
from . import loadcrypto as lc
from . import featbuild as fb
from . import timeutils as tu

# Define the Sats2Trade class that inherits from CryptoData and Candles
class Sats2Trade(lc.CryptoData, fb.Candles):
    def __init__(self, crypto_pair="BTCUSDT", time_frame="15m", market = "spot"):
        # Call the __init__ methods of the parent classes explicitly
        self.time_frame = time_frame
        self.check_data_folder_and_asset_details()
        lc.CryptoData.__init__(self, self.asset_details, self.data_folder, market = market)

        self.trade_time_units(dt=60, kline_size=self.time_frame,
                               starting_date="1 Mar 2022")
        
        self.reference_currency = "USDT"

        if self.reference_currency not in crypto_pair:
            raise ValueError(f"The crypto pair must be priced in the reference currency: \
                              {self.reference_currency}")

        self.crypto_pair = crypto_pair
        self.crypto_asset = crypto_pair.replace(self.reference_currency, "")

        self.assets_to_trade = [self.reference_currency] + [self.crypto_asset]

        cryptoname = self.crypto_pair_dict[crypto_pair]
        fb.Candles.__init__(self, cryptoname, target="UpDown", rollwindow=10)

    def set_data_folder_and_asset_details(self):

        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_dir, "..")
        self.data_folder = os.path.join(root_dir, "data")
        self.asset_details = pd.read_csv(os.path.join(root_dir, "data", "asset_details.csv"))

    # check if there is a folder named data and inside it a file named asset_details.csv
    def check_data_folder_and_asset_details(self):
        if not os.path.isdir("data"):
            raise ValueError(f"data folder does not exist.")
        self.data_folder = "data"
        if not os.path.isfile(os.path.join(self.data_folder, "asset_details.csv")):
            raise ValueError(f"Asset details file {os.path.join(self.data_folder, 'asset_details.csv')} does not exist.")
        self.asset_details = pd.read_csv(os.path.join(self.data_folder, "asset_details.csv"))

    # Function to place an order
    def open_position(self, quantity, order_side="BUY",
                      market = "spot",
                      is_isolated_margin="FALSE"):

    # To try out for a stop loss order
    # order = client.create_order(
    #     symbol = symbol, 
    #     side = SIDE_BUY, 
    #     type = ORDER_TYPE_STOP_LOSS_LIMIT, 
    #     timeInForce = TIME_IN_FORCE_GTC, 
    #     quantity = quantity, 
    #     price = price, 
    #     stopPrice = stopPrice)
        
        if market == "spot":
            try:
                order = self.binance_client.create_order(
                    symbol=self.crypto_pair,
                    side=order_side,
                    type='MARKET',
                    quantity=quantity
                )
                logging.info(f"Spot {order_side} order {order['orderId']} placed successfully: {order}")
            except Exception as e:
                logging.info(f"Error opening Spot {order_side} order:", e)
        elif market == "futures":
            try:
                order = self.binance_client.futures_create_order(
                    symbol=self.crypto_pair,
                    side=order_side,
                    type='MARKET',
                    quantity=quantity,
                    isIsolated=is_isolated_margin
                )
                logging.info(f"Futures {order_side} order {order['orderId']} placed successfully: {order}")
            except Exception as e:
                logging.info(f"Error opening Futures {order_side} order:", e)

    def close_position(self, quantity, order_side = "SELL",
                       market = "spot"):
        
        if market == "spot":
            try:
                order = self.binance_client.create_order(
                    symbol=self.crypto_pair,
                    side=order_side,  # To close a short position, you buy the same amount of the asset
                    type='MARKET',
                    quantity=quantity
                )
                if order_side == "SELL":
                    position_type = "LONG"
                elif order_side == "BUY":
                    position_type = "SHORT"
                logging.info(f"Spot {position_type} position closed successfully: {order}")
            except Exception as e:
                logging.error(f"Error closing Spot {position_type} position:", e)
        elif market == "futures":
            try:
                order = self.binance_client.futures_create_order(
                    symbol=self.crypto_pair,
                    side=order_side,  # To close a short position, you buy the same amount of the asset
                    type='MARKET',
                    quantity=quantity
                )
                if order_side == "SELL":
                    position_type = "LONG"
                elif order_side == "BUY":
                    position_type = "SHORT"
                logging.info(f"Futures {position_type} position closed successfully: {order}")
            except Exception as e:
                logging.error(f"Error closing Futures {position_type} position:", e)



    def close_all_positions(self, market = "spot"):

        if market == "spot":
            try:
                # Get the account information
                account_info = self.binance_client.get_account()
                open_positions = account_info["balances"]
                
                # Loop through the open positions and close them
                for position in open_positions:
                    if position["asset"] == self.crypto_asset:
                        # Check the position side (BUY or SELL) and call the appropriate function to close the position
                        if float(position["free"]) > 1e-5 :
                            self.close_position(float(position["free"]), order_side="SELL", market = market)
            except Exception as e:
                logging.error("Error closing open positions:", e)
  
        elif market == "futures":
            try:
                # Get the account information
                account_info = self.binance_client.futures_account(version=2)
                open_positions = account_info["assets"]
                
                # Loop through the open positions and close them
                for position in open_positions:
                     if position["asset"] == self.crypto_asset:
                #         # Check the position side (BUY or SELL) and call the appropriate function to close the position
                         if float(position["walletBalance"]) > 1e-5 :
                            self.close_position(float(position["free"]), order_side="SELL", market = market)
            except Exception as e:
                logging.error("Error closing open positions:", e)
        #sleep a little bit after closing the positions to make sure the orders are filled  
        time.sleep(10)          
    

    # # Function to place a sell order
    # def place_sell_order(self, quantity):
    #     try:
    #         order = self.binance_client.create_order(
    #             symbol=self.crypto_pair,
    #             side='SELL',
    #             type='MARKET',
    #             quantity=quantity
    #         )
    #         logging.info(f"Sell order {order['orderId']} placed successfully: {order} at {tu.get_utc_timestamp()}")
    #     except Exception as e:
    #         logging.info("Error placing sell order:", e)

    # # Function to place a short sell order
    # def place_short_sell_order(self, quantity):
    #     try:
    #         order = self.binance_client.create_order(
    #             symbol=self.crypto_pair,
    #             side='SELL',
    #             type='MARKET',
    #             quantity=quantity,
    #             isIsolatedMargin='TRUE'  # Enable this if using isolated margin for short selling
    #         )
    #         logging.info(f"Short sell order {order['orderId']} placed successfully: {order} at {tu.get_utc_timestamp()}")
    #     except Exception as e:
    #         logging.info("Error placing short sell order:", e)

    # Function to update the technical indicators and signals
    def update_trade_signal(self):

        self.buildfeatures()
        # Get the signals DataFrame
        in_step = self.rollwindow
        last_step = self.candles.shape[0]
        signals_df = self.get_vma_dataframe_dbg(in_step, last_step)
        if not os.path.isfile("signals.csv"):
            signals_df.to_csv('signals.csv', mode='a', header=True, index=False)
        else:
            signals_df.iloc[[-1]].to_csv('signals.csv', mode='a', header=False, index=False)
        return signals_df        

    def get_portfolio(self, balances, model_signal = 0, 
                        model_current_returns = 0,
                        model_cumulative_returns = 0):
        
        # Calculate the total portfolio value in USDT
        total_value = 0
        for asset in balances["Asset"].values:
            if asset == self.reference_currency :
                total_value += float(balances[balances["Asset"] == asset]['Amount'].values) 
            else:
                asset_symbol = asset + self.reference_currency
                asset_price = self.get_last_price(asset_symbol)
                if asset_price is not None:
                    total_value += (float(balances[balances["Asset"] == asset]['Amount'].values)) * asset_price

        # Calculate the current returns and cumulative returns
        if not hasattr(self, 'initial_balance'):
            self.initial_balance = total_value

        current_returns = (total_value / self.initial_balance) - 1
        if not hasattr(self, 'cumulative_returns'):
            cumulative_returns = 0.0
        else:
            cumulative_returns = (1 + self.cumulative_returns) * (1 + current_returns) - 1
        self.cumulative_returns = cumulative_returns

        # If portfolio_df is not defined, create it
        if not hasattr(self, 'portfolio_df'):
            self.portfolio_df = pd.DataFrame(columns=["Date", "Total Balance",
                                                       "Returns", "Cumulative Returns",
                                                       "Model Returns", "Cumulative Model Returns",
                                                       "Model Signal"])           
            self.portfolio_df.to_csv('portfolio.csv', header=True, index=False)

        # Append the new data to the portfolio_df
        else:
            self.portfolio_df = self.portfolio_df.append(
            {
                "Date": pd.Timestamp.utcnow(),
                "Total Balance": total_value,
                "Returns": current_returns,
                "Cumulative Returns": cumulative_returns,
                "Model Returns": model_current_returns,
                "Cumulative Model Returns": model_cumulative_returns,
                "Model Signal": model_signal,
            },
            ignore_index=True,
        )

            self.portfolio_df.iloc[[-1]].to_csv('portfolio.csv', mode='a', header=False, index=False)

        return total_value, current_returns, cumulative_returns
    
    def get_balance_history(self, balances):
        # If balance_history is not defined, create it
        if not hasattr(self, 'balance_history'):
            self.balance_history = pd.DataFrame(columns=["Date"] + balances["Asset"].values.tolist())
            self.balance_history.to_csv('balance_history.csv', header=True, index=False)

        # Append the new data to the balance_history
        self.balance_history = self.balance_history.append(
                {
                    "Date": pd.Timestamp.now(),
                    self.reference_currency: float(balances[balances["Asset"] == self.reference_currency]['Amount'].values),
                    self.crypto_asset : float(balances[balances["Asset"] == self.crypto_asset]['Amount'].values),
                },
                ignore_index=True,
            )

        self.balance_history.iloc[[-1]].to_csv('balance_history.csv', mode='a', header=False, index=False)

        return self.balance_history

    def trade_loop(self, quantity):
        iteration = 0
        previous_signal = 0
        current_position = 0
        balances = self.get_account_balance(self.assets_to_trade, market = self.market)
        balance_history = self.get_balance_history(balances)
#        balances.to_csv(f'account_balance_{iteration}.csv', header=True, index=False)
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
                logging.info(f"Last signal: {last_signal}")
                # Append signals_df to signals.csv
                signals_df.to_csv('signals.csv', header=True, index=False)
                model_current_returns = signals_df.iloc[-1]['Returns']
                model_cumulative_returns = signals_df.iloc[-1]['Cumulative Returns']

                # If the signal changes from 0 to 1, place a buy order
                if previous_signal == 0 and last_signal == 1:
                    self.open_position(quantity, order_side='BUY', market = self.market,
                                        is_isolated_margin = "FALSE")
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance -= order_value
                    current_position = 1
                    logging.info(f"Placed a buy order for {quantity} {self.crypto_pair} at {tu.get_utc_timestamp()}")
                    logging.info(f"Order value in {self.reference_currency}: {order_value}")

                # If the signal changes from 1 to 0 and there is an open position, close the position
                elif previous_signal == 1 and last_signal == 0 and current_position == 1:
                    self.close_position(quantity, order_side='SELL', market = self.market)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance += order_value
                    current_position = 0
                    logging.info(f"Closed position for {quantity} {self.crypto_pair} at {tu.get_utc_timestamp()}")
                    logging.info(f"Order value in {self.reference_currency}: {order_value}")                    

                # If the signal changes from 0 to -1, place a short sell order
                elif previous_signal == 0 and last_signal == -1:
                    self.open_position(quantity, order_side='SELL', market=self.market,
                                       is_isolated_margin = "TRUE")
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance -= order_value
                    current_position = -1
                    logging.info(f"Placed a short sell order for {quantity} {self.crypto_pair} at {tu.get_utc_timestamp()}")
                    logging.info(f"Order value in {self.reference_currency}: {order_value}")

                # If the signal changes from -1 to 0 and there is an open short position, close the short position
                elif previous_signal == -1 and last_signal == 0 and current_position == -1:
                    self.close_position(quantity, order_side='BUY', market = self.market)
                    # Calculate the value of the order in USDT and update the balance
                    order_value = quantity * self.get_last_price(self.crypto_pair)
                    self.initial_balance += order_value
                    current_position = 0
                    logging.info(f"Closed short position for {quantity} {self.crypto_pair} at {tu.get_utc_timestamp()}")
                    logging.info(f"Order value in {self.reference_currency}: {order_value}")
                else:
                    logging.info("No trade action taken.")

                # Calculate the portfolio details
                balances = self.get_account_balance(self.assets_to_trade, market = self.market)
                balance_history = self.get_balance_history(balances)
                total_balance, current_returns, cumulative_returns = self.get_portfolio(balances,
                                                                                         model_signal=last_signal,
                                                                                         model_current_returns=model_current_returns,
                                                                                         model_cumulative_returns=model_cumulative_returns)
                logging.info("Portfolio DataFrame:")
                logging.info(f"Total Balance: {total_balance}")
                logging.info(f"Current Returns: {current_returns}")
                logging.info(f"Cumulative Returns: {cumulative_returns}")
                iteration += 1
                #balances.to_csv(f'account_balance_{iteration}.csv', header=True, index=False)

                # Set the previous signal for the next iteration
                previous_signal = last_signal
                # Update the current position based on the signal
                current_position = last_signal

                # Sleep for a certain period before the next iteration
                #time.sleep(60)  # Sleep for 60 seconds (adjust this based on your strategy)
            #except Exception as e:
            #    print("Error occurred:", e)
