import argparse
from sats4u.sats2trade import Sats2Trade 

def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the trading algorithm.")
    
    # Add the arguments as options
    parser.add_argument("--quantity", type=float, help="Quantity value for the trades.")
    parser.add_argument("--crypto_pair", type=str, help="Cryptocurrency pair for trading.")
    parser.add_argument("--time_frame", type=str, choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"], 
                        help="Possible time-frame to trade, i.e. candlestick time duration.")
    parser.add_argument("--testnet", action="store_true", help="Use testnet mode for trading.")
    parser.add_argument("--secrets_filename", type=str, help="Path to the secrets filename.")
    
    # Parse the arguments
    args = parser.parse_args()

    return args

def run_trade():
    # Parse the arguments
    args = parse_arguments()
    # Retrieve the values from the parsed arguments
    quantity_val = args.quantity
    crypto_pair_val = args.crypto_pair
    time_frame_val = args.time_frame
    testnet_val = args.testnet
    secrets_filename = args.secrets_filename

    sats2trade = Sats2Trade(crypto_pair=crypto_pair_val, time_frame = time_frame_val)
    # sats2trade.set_binance_api_keys(api_key, api_secret, server_location='not-US', testnet=True)
    sats2trade.load_binance_client(secrets_filename, testnet=testnet_val)
    # Start the algorithmic trading loop
    sats2trade.trade_loop(quantity_val)

if __name__ == "__main__":
    run_trade()