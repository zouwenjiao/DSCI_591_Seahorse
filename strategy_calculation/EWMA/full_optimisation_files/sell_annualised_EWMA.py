import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import OrderedDict
#import talib as ta
from os import listdir
from os.path import isfile, join
import pickle

from tqdm import tqdm
# Remove settingwithcopy warning
pd.options.mode.chained_assignment = None


def opt(data_path, trade_signals, ewma_window = 5):
    df = pd.read_csv(data_path, delimiter=r"\s+",header=None)
    df.columns = ["date", "price", "oscillator"]
    df["date"] = df["date"].apply(lambda x: datetime.strptime(str(x), "1%y%m%d.00%H%M.00"))
    df = df.reset_index()
    df = df.rename(columns = {"index":"interval_index"})

    ##############################################################
    # Filter 3-2 (EWMA-X with slope value threshold )
    ##############################################################
    # Calculating exponential moving average X = 5
    df["ewma_price"] = df["price"].ewm(span = ewma_window,
                                       min_periods = 0).mean()

    # ewma_price gradient for Momentum Filter
    df["ewma_price_gradient"] = df["ewma_price"] - df["ewma_price"].shift(1)
    df["ewma_price_gradient_percent"] = df["ewma_price_gradient"] /df["ewma_price"].shift(1)

    ##############################################################
    # Identifying primary oscillator signals
    ##############################################################
    # find lags of oscillator to obtain peaks/troughs pattern
    df["oscillator_lag1"] = df.oscillator.shift(1)
    df["oscillator_lag2"] = df.oscillator.shift(2)

    condlist = [
        # Buy signal directly after a direct trough: (Low, Lowest, Low) OR (Lowest, Lowest, Low)
        (df["oscillator"] > df["oscillator_lag1"]) & (df["oscillator_lag2"] >= df["oscillator_lag1"]),

        # Sell signal after a direct peak: (High, Highest, High) OR (Highest, Highest, High)
        (df["oscillator"] < df["oscillator_lag1"]) & (df["oscillator_lag2"] <= df["oscillator_lag1"])
    ]

    # What to return for "Buy" or "Sell" based on the order in condlist
    choicelist = [1,-1]

    df["action"] = np.select(condlist, choicelist)

    # Find all primary signals. However, there might still be consecutive same signals in a sequence of plateau
    df_signal = df[df["action"] != 0]

    # Create action_lag1 to verify if there are consecutive same signals
    df_signal["action_lag1"] = df_signal["action"].shift(1)

    # action_joint should be -1 if previous signal is not the same. ie (Buy then Sell = 1 * -1 = -1)
    df_signal["action_joint"] = df_signal["action"] * df_signal["action_lag1"]

    # Filter out all consecutive signals.
    # Note that the very 1st signal row has NaN for action_joint due to lag but we can keep it
    df_signal = df_signal[df_signal["action_joint"] != 1].reset_index(drop=True)

    ##############################################################
    # Calculating holding time AND Percentage returns of each PRIMARY trade signal
    ##############################################################
    # Finding holding time by shifting date of next signal
    df_signal["exit_date"] = df_signal["date"].shift(-1)
    df_signal["holding_time"] = df_signal["exit_date"] - df_signal["date"]

    # Convert holding time to days float type
    df_signal["holding_time_intervals"] = df_signal["interval_index"].shift(-1) - df_signal["interval_index"]

    # note that last row has a NAN because it does not have an exit trade
    # Keep it for now temporarily for % return calculation more wrangling

    # Find returns in terms of pct_return and overall portfolio_change
    df_signal["exit_signal_percent_change"] = df_signal["price"].pct_change().shift(-1)
    df_signal["portfolio_change"] = df_signal["exit_signal_percent_change"] * df_signal["action"] + 1
    df_signal["percent_return"] = df_signal["exit_signal_percent_change"] * df_signal["action"]


    ##############################################################
    # Clean up dataframe
    ##############################################################
    # Remove unnecessary columns
    df_signal = df_signal.drop(["exit_date", "holding_time", "oscillator_lag1", "oscillator_lag2", "action_lag1", "action_joint"], axis = 1)

    # Drop last signal because it does not have an exit trade signal
    df_signal = df_signal[:-1]

    ##############################################################
    # Filter 1+2+3-2(EWMA-X slope threshold value)
    ##############################################################

    df_signal["ewma_price_gradient_percent_action"] = df_signal["ewma_price_gradient_percent"]* df_signal["action"]

    # Filter 1 + 2 + +3: Appropriate osc polarity & no trading if abs(osc) > 7 & Momentum Filter
    filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) &\
                (abs(df_signal["oscillator"]) < 7) &\
                (df_signal["ewma_price_gradient_percent_action"] > 0)

    filtered_signal = df_signal[filter_boolean]

    ##############################################################
    # Check for trade signals
    ##############################################################
    # Buy signals only
    if trade_signals == "buy_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==1]
    # Sell signals only
    elif trade_signals == "sell_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==-1]
    # Both signals
    else:
        filtered_signal= filtered_signal

    print("Number of '{0}' filtered signals = {1}".format(trade_signals.upper(), len(filtered_signal)))

       # Extract year from datetime for aggregation calculation
    filtered_signal["year"] = filtered_signal["date"].dt.year

    # Creating stock ticker by doing chained splitting
    filtered_signal['stock']= str(data_path).split("/")[-1].split("2")[0]

    return filtered_signal


directory_path = '../data/2020/'
files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]


def all_stocks(ma, action):
    storage = []
    for i in range(len(files_list)):
        data_path = "../../../data/2020/" + files_list[i]
        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            df = opt(data_path, action, ewma_window = ma)
            returns = df.groupby(["year","stock"]).sum()["percent_return"]
            time_interval = df.groupby(["year","stock"]).sum()["holding_time_intervals"]
            metric = (returns/time_interval) * 252 * 13
            storage.append(metric)

    # Convert the storage list into a dataframe and take the average of all stocks in list

    m = pd.concat(storage).groupby('year')
    return m.mean()


def backtest_ma(ma_list, strat_type):
    storage = []
    for ma in ma_list:
        print(ma)
        storage.append(all_stocks(ma, strat_type))
    return storage
    df = pd.DataFrame(storage).T


ma_list = range(1, 100)
z = backtest_ma(ma_list, 'sell_only')
results = pd.DataFrame(z).T
results.columns = ma_list
results.columns = ['ewma_'+ str(col) for col in results.columns]
results.index = list(range(2000, 2021))
results.to_csv("sell_annualized_ewma.csv")
