import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import OrderedDict
import talib as ta
from os import listdir
from os.path import isfile, join
import pickle

from tqdm import tqdm
# Remove settingwithcopy warning
pd.options.mode.chained_assignment = None  

def filtered_signals_rsi(trade_signals, data_path, lookback_step = 6, upper_threshold=60, lower_threshold=40):
    """
    Generates the filter signals for the RSI strategy given input parameters for a given stock.
    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    data_path : string
        String input for relative data path location.
    rsi_window : int (default=6)
        Parameter value for relative strength indicator operation. Represents the intra-day intervals to be used.
    upper_threshold: int (default=60)
        The upper threshold of rsi value, above which the stock is thought to be over-bought, or overvalued.
    lower_threshold: int (default=40)
        The lower threshold of rsi value, below which the stock is thought to be over-sold, or undervalued.
            

    
    Returns
    -------
    pandas.DataFrame object
        Dataframe object containing all filtered trade signals and relevant variables like stock price, oscillator value, percent return, etc.
    """
    ##############################################################
    # Reading in data
    ##############################################################
    df = pd.read_csv(data_path, delimiter=r"\s+",header=None)
    df.columns = ["date", "price", "oscillator"]
    df["date"] = df["date"].apply(lambda x: datetime.strptime(str(x), "1%y%m%d.00%H%M.00"))

    
    # Reset index for time intervals
    df = df.reset_index()
    df = df.rename(columns = {"index":"interval_index"})
    
    # Calculating RSI 
    df["rsi"] = ta.RSI(df['price'], timeperiod = lookback_step)
    
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
        
    ##############################################################
    # Calculating holding time AND Percentage returns of each PRIMARY trade signal
    ##############################################################
    # Convert holding time to days float type
    df_signal["holding_time_intervals"] = df_signal["interval_index"].shift(-1) - df_signal["interval_index"]

    # note that last row has a NAN because it does not have an exit trade
    # Keep it for now temporarily for % return calculation more wrangling

    # Find returns in terms of pct_return and overall portfolio_change
    df_signal["exit_signal_percent_change"] = df_signal["price"].pct_change().shift(-1)
    df_signal["portfolio_change"] = df_signal["exit_signal_percent_change"] * df_signal["action"] + 1
    df_signal["percent_return"] = df_signal["exit_signal_percent_change"] * df_signal["action"] 
    
    
    # Remove unnecessary columns
    df_signal = df_signal.drop(["oscillator_lag1", "oscillator_lag2", "action_lag1", "action_joint"], axis = 1)

    # Drop last signal because it does not have an exit trade signal
    df_signal = df_signal[:-1]

    # Filter 1 + 2 + +3: Appropriate osc polarity & no trading if abs(osc) > 7 & Momentum Filter
    filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) &\
                (abs(df_signal["oscillator"]) < 7) & \
                (((df_signal["action"]*df_signal["rsi"] < lower_threshold) & (df_signal["action"]*df_signal["rsi"] > 0)) | \
                (df_signal["action"]*df_signal["rsi"] < -upper_threshold))
    
    filtered_signal = df_signal[filter_boolean]
    if trade_signals == "buy_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==1]
    # Sell signals only
    elif trade_signals == "sell_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==-1]
    # Both signals
    else:
        filtered_signal = filtered_signal
    ##############################################################
    # Extract year and month for aggregation basis
    ##############################################################
    # Extract year from datetime for aggregation calculation
    filtered_signal["year"] = filtered_signal["date"].dt.year

    # Creating stock ticker by doing chained splitting 
    filtered_signal['stock']= str(data_path).split("/")[-1].split("2")[0]
    
    # returns df
    return filtered_signal

# directory paths
directory_path = "../../../data/2020/"

files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

###########################################################################
def portfolio_eval(rsi, action, upper=60, lower=40):
    """
    Compiles the filter signals for the RSI strategy given input parameters for all stocks, and calculates the 
    average annualised return metric on a yearly breakdown assuming equal stock weightage basket. 
    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    directory_path : string
        String input for relative directory path location where all stocks data are kept.
    rsi_window : int (default=6)
        Parameter value for relative strength indicator operation. Represents the intra-day intervals to be used.
    upper_threshold: int (default=60)
        The upper threshold of rsi value, above which the stock is thought to be over-bought, or overvalued.
    lower_threshold: int (default=40)
        The lower threshold of rsi value, below which the stock is thought to be over-sold, or undervalued.
    
    Returns
    -------
    pandas.Series object
        Series object with yearly breakdown of average annualised return metric. 
    """
    
    for i in range(len(files_list)):
        data_path = directory_path + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            if i == 0:
                compiled_df = filtered_signals_rsi(action, data_path, 
                                                  lookback_step = rsi, 
                                                  upper_threshold=upper,
                                                  lower_threshold=lower)
                

                
            else:
                intermediate_df = filtered_signals_rsi(action, data_path, 
                                                      lookback_step = rsi, 
                                                      upper_threshold=upper,
                                                      lower_threshold=lower)
                
                compiled_df = pd.concat([compiled_df, intermediate_df])
                
    
    # Applying formula in a quick manner (*multiply by 252 * 13 as per colin's email) using stock and year
    metric_year_stock = compiled_df.groupby(["year","stock"]).sum()["percent_return"]/compiled_df.groupby(["year","stock"]).sum()["holding_time_intervals"] * 252 * 13
        
    # finding the mean for yearly performance with equal weightage stock basket
    metric_year = metric_year_stock.groupby("year").mean()
    
    return metric_year

def get_table(rsi_lookback, type_action):
    '''
    Function for creating column of '% Ret/ Holding Time'
    '''

    avg_return = []
    idx = []
    
    df = pd.DataFrame()
    for day in tqdm(rsi_lookback):
        df_all = portfolio_eval(day, type_action, upper=60, lower=40)
        index = "rsi_"+ str(day)
        df[index] = df_all
    
    return df

###########################################################################
# RSI search range
rsi_search = np.arange(2,101,1)

buy_df = get_table(rsi_search, 'buy_only')

buy_df.to_csv("buy_df.csv")

print("Buy optimisation completed!")