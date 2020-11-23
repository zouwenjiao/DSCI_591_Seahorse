import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import talib as ta
from os import listdir
from os.path import isfile, join
import pickle

from tqdm import tqdm
# Remove settingwithcopy warning
pd.options.mode.chained_assignment = None  

def filter_signals_MA_MACD(trade_signals, data_path, ma_window, mfast, mslow, msignal):
    """
    Generates the filter signals for the MA-MACD strategy given input parameters for a given stock.

    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    data_path : string
        String input for relative data path location.
    ma_window : int
        Parameter value for moving average operation. Represents the intra-day intervals to be used. 
    mfast : int
        Parameter value for MACD Fast exponentially weighted moving average series. Represents day counts and not intra-day intervals.
    mslow : int
        Parameter value for MACD Slow exponentially weighted moving average series.  Represents day counts and not intra-day intervals.
    msignal: int
        Parameter value for MACD Signal exponentially weighted moving average to be applied on the MACD histogram. Represents day counts and not intra-day intervals.
    
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
    
    ##############################################################
    # MA-MACD
    ##############################################################
    df["ma_price"] = df["price"].rolling(window = ma_window).mean()
    _, _, df['macd_hist'] = ta.MACD(df.price, fastperiod=mfast*13, slowperiod=mslow*13, signalperiod=msignal*13)

    # Calculate Gradients For Momentum Indicators
    df["ma_price_gradient"] = df["ma_price"] - df["ma_price"].shift(1)
    #df['macd_gradient'] = df['macd_hist'] - df['macd_hist'].shift(1)
    
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
    df_signal["holding_time_in_days"] = df_signal["holding_time"]/pd.Timedelta(1.0,unit='D')
    
    
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
    # Filter 1+2+3(MA)+4(MACD)
    ##############################################################
    # Filter 1 + 2 + +3: Appropriate osc polarity & no trading if abs(osc) > 7 & Momentum Filter
    filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) &\
                     (abs(df_signal["oscillator"]) < 7) &\
                   (df_signal["action"]*df_signal["macd_hist"] > 0) &\
                    (df_signal["action"]*df_signal["ma_price_gradient"] > 0)
    
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

    ##############################################################
    # Extract year and month for aggregation basis
    ##############################################################
    # Extract year from datetime for aggregation calculation
    filtered_signal["year"] = filtered_signal["date"].dt.year
    
    # Calculate cumprod for whole dataset
    filtered_signal["cumprod_portfolio_change"] = filtered_signal["portfolio_change"].cumprod()
    
    # Creating stock ticker by doing chained splitting 
    filtered_signal['stock']= str(data_path).split("/")[-1].split("2")[0]

    return filtered_signal

###########################################################################

# Portfolio evaluation function
def portfolio_eval_ma_macd(trade_signals, directory_path, ma_window, mfast, mslow, msignal):
    """
    Compiles the filter signals for the MA-MACD strategy given input parameters for all stocks, and calculates the 
    average annualised return metric on a yearly breakdown assuming equal stock weightage basket. 

    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    directory_path : string
        String input for relative directory path location where all stocks data are kept.
    ma_window : int
        Parameter value for moving average operation. Represents the intra-day intervals to be used. 
    mfast : int
        Parameter value for MACD Fast exponentially weighted moving average series. Represents day counts and not intra-day intervals.
    mslow : int
        Parameter value for MACD Slow exponentially weighted moving average series.  Represents day counts and not intra-day intervals.
    msignal: int
        Parameter value for MACD Signal exponentially weighted moving average to be applied on the MACD histogram. Represents day counts and not intra-day intervals.
    
    Returns
    -------
    pandas.Series object
        Series object with yearly breakdown of average annualised return metric. 
    """
    # Create files_list based on directory of train or test path
    files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    total_count_primary_signals = 0
    total_count_filtered_signals = 0
    

    for i in range(len(files_list)):
        data_path = directory_path + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            if i == 0:
                compiled_df = filter_signals_MA_MACD(trade_signals, data_path, ma_window, mfast, mslow, msignal)
                
            else:
                intermediate_df = filter_signals_MA_MACD(trade_signals, data_path, ma_window, mfast, mslow, msignal)
                
                compiled_df = pd.concat([compiled_df, intermediate_df])
    
    # Applying formula in a quick manner (*multiply by 252 * 13 as per colin's email) using stock and year
    metric_year_stock = compiled_df.groupby(["year","stock"]).sum()["percent_return"]/compiled_df.groupby(["year","stock"]).sum()["holding_time_intervals"] * 252 * 13
    
    # finding the mean for yearly performance with equal weightage stock basket
    metric_year = metric_year_stock.groupby("year").mean()
    
    return metric_year

###########################################################################
def strategy_eval_ma_macd(trade_signals, directory_path, ma_params_list, macd_params_dict):
    """
    Compiles the average annualised return metric on a yearly breakdown assuming equal stock weightage basket for multiple MA-MACD parameter combinations.

    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    directory_path : string
        String input for relative directory path location where all stocks data are kept.
    ma_params_list: list
        List of MA window parameters (integer) for moving average operation. Represents the intra-day intervals to be used. 
    macd_params_dict : dict
        Dictionary of MACD parameters for different MACD combinations.
    
    Returns
    -------
    pandas.DataFrame object
        DataFrame object with yearly breakdown of average annualised return metric for multiple MA-MACD parameter combinations.
    """
    # Dictionary for storing data
    strategy_yearly_results = {}
    
    for i in tqdm(range(len(ma_params_list))):
        ma_window = ma_params_list[i]
        
        for macd_combi in range(len(macd_params_dict)):
            # Create strategy_name for key
            strategy_name = "ma("+str(ma_window) + "/13)_macd("+str(macd_params_dict[macd_combi]["mfast"]) + ","+str(macd_params_dict[macd_combi]["mslow"])+","+str(macd_params_dict[macd_combi]["msignal"])+")"
            
            # Store yearly results 
            strategy_yearly_results[strategy_name] = portfolio_eval_ma_macd(trade_signals, directory_path, ma_window, 
                                                                             macd_params_dict[macd_combi]["mfast"], 
                                                                             macd_params_dict[macd_combi]["mslow"], 
                                                                             macd_params_dict[macd_combi]["msignal"])
    return pd.DataFrame(strategy_yearly_results)

###########################################################################

def strategy_statistics_MA_MACD(trade_signals, directory_path, ma_window, mfast, mslow, msignal):
    """
    Calculate relevant statistics for a given MA-MACD strategy.

    Parameters:
    -----------
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    directory_path : string
        String input for relative directory path location where all stocks data are kept.
    ma_window : int
        Parameter value for moving average operation. Represents the intra-day intervals to be used. 
    mfast : int
        Parameter value for MACD Fast exponentially weighted moving average series. Represents day counts and not intra-day intervals.
    mslow : int
        Parameter value for MACD Slow exponentially weighted moving average series.  Represents day counts and not intra-day intervals.
    msignal: int
        Parameter value for MACD Signal exponentially weighted moving average to be applied on the MACD histogram. Represents day counts and not intra-day intervals.
    
    Returns
    -------
    tuple
        Returns a tuple of 3 elements that represent the filter trade signal counts, the average portfolio cumulative return, and the average holding time per signal.
    """
    # Create files_list based on directory of train or test path
    files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    total_count_primary_signals = 0
    total_count_filtered_signals = 0
    
    recession_years = [2000, 2008, 2020]
    
    for i in range(len(files_list)):
        data_path = directory_path + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            if i == 0:
                compiled_df = filter_signals_MA_MACD(trade_signals, data_path, ma_window, mfast, mslow, msignal)
                
            else:
                intermediate_df = filter_signals_MA_MACD(trade_signals, data_path, ma_window, mfast, mslow, msignal)
                
                compiled_df = pd.concat([compiled_df, intermediate_df])
                    
    # Summing all filtered signals for all stocks for 20 years for this strategy
    filter_counts = len(compiled_df)
    
    # Finding the equal-weightage stock basket cumulative returns
    avg_cum_ret_by_equal_stock_basket = compiled_df.groupby("stock").tail(1)["cumprod_portfolio_change"].mean()
    
    # Average of holding time in days
    avg_holding_time_per_signal = compiled_df.holding_time_in_days.sum()/filter_counts
    
    return (filter_counts, avg_cum_ret_by_equal_stock_basket, avg_holding_time_per_signal)

###########################################################################

def get_stats(data, trade_signals, directory_path):
    """
    Given the data of yearly average annualised return metric for multiple strategies, obtain the best strategy combination based on 
    the average annualised return, and calculate the relevant statistics.

    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame of yearly average annualised return metric for multiple strategies
    trade_signals : string
        String input for specific trade signals . Accepts "buy_only" or "sell_only". If it is not in either, it will generate both Buy and Sell signals.
    directory_path : string
        String input for relative directory path location where all stocks data are kept.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame object with the statistics of the best strategy combination.
    """
    # Get top avg metric strategies 
    total_mean_metric = data.mean().sort_values(ascending=False).head(1)
    top_filters = total_mean_metric.index.values
    total_std_metric = data[top_filters].std()

    total_mean_per_std_metric = total_mean_metric/total_std_metric

    # Compile into DataFrame
    col = [total_mean_metric, total_std_metric, total_mean_per_std_metric]
    col_names = ['avg_metric', 'std_metric', "avg_metric/std_metric"]
    df = pd.DataFrame(col).T
    df.columns = col_names

    # Strategy statistics
    filter_counts_list =[]
    avg_cum_ret_list = []
    avg_holding_time_per_signal_list = []
    
    top_strategy_name_list = list(total_mean_metric.index)

    for i in tqdm(range(len(top_strategy_name_list))):

        # MA window
        ma_window = int(top_strategy_name_list[i].split("/")[0].split("(")[-1])

        # MACD params 
        # returns a list of example [`12`, `26`, `9`]. note that they are string elements
        macd_combi = top_strategy_name_list[i].split("/")[1].split("(")[-1].split(")")[0].split(",")

        mfast = int(macd_combi[0])
        mslow = int(macd_combi[1])
        msignal = int(macd_combi[2])

        # Calculate strategy statistics
        filter_counts, avg_cum_ret_by_equal_stock_basket, avg_holding_time_per_signal \
        = strategy_statistics_MA_MACD(trade_signals, directory_path, 
                                      ma_window,
                                      mfast,
                                      mslow,
                                      msignal)
        
        filter_counts_list.append(filter_counts)
        avg_cum_ret_list.append(avg_cum_ret_by_equal_stock_basket)
        avg_holding_time_per_signal_list.append(avg_holding_time_per_signal)

    # Compile into DataFrame
    stats_df = pd.DataFrame([filter_counts_list, avg_cum_ret_list, avg_holding_time_per_signal_list]).T
    stats_df.index = total_mean_metric.index
    stats_df.columns = ["filter_counts", "avg_port_cum_return","avg_holding_time_per_signal"]

    compiled_results = pd.concat([df, stats_df] ,axis = 1)

    return compiled_results