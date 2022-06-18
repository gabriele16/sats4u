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



def backtest(df_preds_true, short_or_long = "long", fee=0.025):

    wallet = 0
    total_wallet_history = []
    single_wallet_history = []

    buys_cnt = 0
    buys_cnt_win = 0
    buys_cnt_losses = 0
    drawback = 0
    old_profit_negative = False
    old_profits = 0

    for i in range(split_point, len(true_vals) - step_back):
        predicted_close = preds[i - split_point]
        previous_close = true_vals[i]
        real_next_close = true_vals[i+1]

        if (previous_close + (previous_close * fee)) < predicted_close:  # buy
            profit = real_next_close - previous_close
            if profit > 0:
                profit = profit - (profit * fee)
                buys_cnt_win += 1
                old_profit_negative = False
            else:
                profit = profit + (profit * fee)
                buys_cnt_losses += 1
                if old_profit_negative:
                    old_profits += profit
                else:
                    old_profits = profit
                if old_profits < drawback:
                    drawback = old_profits
                old_profit_negative = True
            wallet += profit
            total_wallet_history.append(wallet)
            single_wallet_history.append(profit)
            buys_cnt += 1
        else:
            old_profit_negative = False
            old_profits = 0

    # print('Fee:', fee)
    # print('----------------------')
    # print('Buy     ', buys_cnt, '(', buys_cnt_win, 'ok', buys_cnt_losses, 'ko )')
    # print('Wallet  ', wallet)
    # print('Drawback', drawback)

    return total_wallet_history, single_wallet_history, wallet