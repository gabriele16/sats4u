import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kelly_exp_returns(returns):

  drift = np.mean(returns)
  variance = np.var(returns)
  kelly_frac = drift/variance
  return kelly_frac

def kelly_exp_simple(pct_gain,pct_loss,n_wins):

  if pct_loss != 0.0 and pct_gain != 0.0:
      win_loss_ratio = pct_gain/pct_loss
      kelly_frac = n_wins - (1- n_wins)/win_loss_ratio
  else:
      kelly_frac = np.nan
  return kelly_frac

def backtest_df(df_preds_true, step_back, long_short = "long", fee=0.025):

    if long_short != "long" and long_short != "short" and long_short != "longshort":
        raise ValueError("Can only have long, short or longshort")

    wallet = 0
    total_wallet_history = []
    single_wallet_history = []
    datetime_iter = []

    buys_cnt = 0
    buys_cnt_win = 0
    buys_cnt_losses = 0
    drawback = 0
    old_profit_negative = False
    old_profits = 0
    mean_pct_gain = 0
    mean_pct_loss = 0

    delta = df_preds_true.index[1]-df_preds_true.index[0]
    df_preds_true.iloc[:,0] = df_preds_true.iloc[:,0].shift(step_back,delta)

    previous_true_close = df_preds_true.iloc[0,0]
    previous_pred_close = df_preds_true.iloc[0,-1]
    it = 1

    for index, row in df_preds_true.iloc[1:].iterrows():
        true_close = row[0]
        pred_close = row[-1]

        if long_short == "long":
            if previous_true_close + previous_true_close*fee < pred_close:  # long
                profit = true_close - previous_true_close
                if profit > 0:
                    profit = profit - (profit * fee)
                    buys_cnt_win += 1
                    old_profit_negative = False
                    mean_pct_gain += (true_close/previous_true_close)                   

                else:
                    profit = profit + (profit * fee)
                    buys_cnt_losses += 1
                    mean_pct_loss += (true_close/previous_true_close)
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

        elif long_short == "short":
            if  previous_true_close*(1+fee) > pred_close:  # short
                profit = -1*(true_close - previous_true_close)
                if profit > 0:
                    # win
                    profit = profit - (profit * fee)
                    buys_cnt_win += 1
                    old_profit_negative = False
                    # if short we gain (i.e. mean_pct_gain >1 ) when prev close > true close
                    mean_pct_gain += (previous_true_close/true_close)             
                else:
                    #loss
                    profit = profit + (profit * fee)
                    buys_cnt_losses += 1
                    #if we short we lose (i.e. mean_pct_loss < 1) when prev close < true close
                    mean_pct_loss += (previous_true_close/true_close)
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
                datetime_iter.append(it)
                buys_cnt += 1
            else:
                old_profit_negative = False
                old_profits = 0

        elif long_short == "longshort":
            if  previous_true_close *(1+fee) > pred_close:  # short
                profit = true_close - previous_true_close
                if profit < 0:
                    # win
                    profit = -1*(profit - (profit * fee))
                    buys_cnt_win += 1
                    old_profit_negative = False
                    # if short we gain (i.e. mean_pct_gain >1 ) when prev close > true close
                    mean_pct_gain += (previous_true_close/true_close)                  
                else:
                    #loss
                    profit = -1*(profit + (profit * fee))
                    buys_cnt_losses += 1
                    #if we short we lose (i.e. mean_pct_loss < 1) when prev close < true close
                    mean_pct_loss += (previous_true_close/true_close)
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
                datetime_iter.append(it)
                buys_cnt += 1
            elif previous_true_close *(1+fee) < pred_close:  # long
                profit = true_close - previous_true_close
                if profit > 0:
                    profit = profit - (profit * fee)
                    buys_cnt_win += 1
                    old_profit_negative = False
                    mean_pct_gain += (true_close/previous_true_close)
                else:
                    profit = profit + (profit * fee)
                    buys_cnt_losses += 1
                    mean_pct_loss += (true_close/previous_true_close)
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
                datetime_iter.append(it)
                buys_cnt += 1

        previous_true_close = true_close
        previous_pred_close = pred_close

    if buys_cnt_win != 0.0:
        mean_pct_gain /= buys_cnt_win
        wins_pct =  buys_cnt_win/buys_cnt
    else:
        wins_pct = 0.0
    if buys_cnt_losses !=  0.0:
        mean_pct_loss /= buys_cnt_losses
    
    print('Fee:', fee)
    print('----------------------')
    print('Buy     ', buys_cnt, '(', buys_cnt_win, 'ok', buys_cnt_losses, 'ko )')
    print('Avg PCT gain:', mean_pct_gain)
    print('Avg PCT loss:', mean_pct_loss)
    print('Wins  PCT  ', wins_pct)
    print('Avg PCT Gain.   ', mean_pct_gain)
    print('No-op   ', buys_cnt - buys_cnt_win - buys_cnt_losses)
    print('Wallet  ', wallet)
    print('Drawback', drawback)

    kelly_frac = kelly_exp_simple(mean_pct_gain,mean_pct_loss,wins_pct)

    print('Kelly Fraction   ',kelly_frac)

    wallet_hist_df = pd.DataFrame(np.array([total_wallet_history, single_wallet_history]).T,
                                            index=df_preds_true.index[datetime_iter],
                                            columns=["Tot. Wallet hist", "Single Wallet hist" ])

    return wallet_hist_df, wallet, kelly_frac


def show_backtest_results(wallet,wallet_hist_df):

    print('Total earned', wallet)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    wallet_hist_df.plot(y= 0 ,ax=axes[0,0])
    wallet_hist_df.plot(y= 1 ,ax=axes[0,1])

    plt.tight_layout()
    plt.show()


def backtest_debug(preds, true_vals, split_point, step_back, fee=0.025):

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

        print('Fee:', fee)
        print('----------------------')
        print('Buy     ', buys_cnt, '(', buys_cnt_win, 'ok', buys_cnt_losses, 'ko )')
        print('Wallet  ', wallet)
        print('Drawback', drawback)

        wallet_hist_df = pd.DataFrame(np.array([total_wallet_history, single_wallet_history]).T,
                                     columns=["Tot. Wallet hist", "Single Wallet hist" ])


        return total_wallet_history, single_wallet_history, wallet