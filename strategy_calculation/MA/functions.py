import pandas as pd
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join
pd.options.mode.chained_assignment = None

def filtered_signal_ma(data_path, trade_signals, ma_window):
    """ 
    Read raw csv data, and split date time. Extract all trade signals 
    from oscillator, and apply basic filters and MA with specified window
    to filter out signals. Return a table containing information 
    about profitability of each trade, as well as holding time.

    Parameters
    ----------
    trade_signals : str
        A string indicate whether we want to look at sell/buy signal only.
    data_path: str
        A string indicate the data user want to read.
    ma_window : int
        A integer indicates size of moving average window
    
    Returns
    -------
    pandas.core.frame.DataFrame 
        A dataframe that contains filtered signal and corresponding information.
        
    Example
    --------
    >>> data_path = ''../../data/2020/''
    >>> filtered_signal_ma('buy_only',data_path,1)
    """
    df = pd.read_csv(data_path, delimiter=r"\s+",header=None)
    df.columns = ["date", "price", "oscillator"]
    df["date"] = df["date"].apply(lambda x: datetime.strptime(str(x), "1%y%m%d.00%H%M.00"))
    df = df.reset_index()
    df = df.rename(columns = {"index":"interval_index"})

    ##############################################################
    # Filter 3-2 (EWMA-X with slope value threshold )
    ##############################################################
    # Calculating exponential moving average X = 5
    df["ma_price"] = df["price"].rolling(window = int(ma_window*13)).mean()
    # ma_price gradient for Momentum Filter
    df["ma_price_gradient"] = df["ma_price"] - df["ma_price"].shift(1)
    df["ma_price_gradient_percent"] = df["ma_price_gradient"] /df["ma_price"].shift(1)

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
    df_signal["holding_time_in_days"] = df_signal["holding_time"]/pd.Timedelta(1.0,unit='D')
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
    df_signal = df_signal.drop(["exit_date", "holding_time", "oscillator_lag1", "oscillator_lag2", "action_lag1",  "action_joint"], axis = 1)

    # Drop last signal because it does not have an exit trade signal
    df_signal = df_signal[:-1]

    ##############################################################
    # Filter 1+2+3-2(MA-X slope threshold value)
    ##############################################################

    df_signal["ma_price_gradient_percent_action"] = df_signal["ma_price_gradient_percent"]* df_signal["action"]

    # Filter 1 + 2 + +3: Appropriate osc polarity & no trading if abs(osc) > 7 & Momentum Filter
    filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) &\
                (abs(df_signal["oscillator"]) < 7) &\
                (df_signal["ma_price_gradient_percent_action"] > 0)

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

    #print("Number of '{0}' filtered signals = {1}".format(trade_signals.upper(), len(filtered_signal)))

       # Extract year from datetime for aggregation calculation
    filtered_signal["year"] = filtered_signal["date"].dt.year
    filtered_signal["cumprod_portfolio_change"] = filtered_signal["portfolio_change"].cumprod()
    # Creating stock ticker by doing chained splitting
    filtered_signal['stock']= str(data_path).split("/")[-1].split("2")[0]

    return filtered_signal

def get_annual(action,ma):
    '''
    Combine all filtered and processed
    stocks data, and calculate the annulized
    return based on specified MA window size for each year.

    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    ma_window : int
        A integer indicates size of moving average window

    Returns
    -------
    pandas.core.series.Series
        A series that contains all annualized return for each year.

    Example
    -------
    >>>get_annual('buy_only',3)
            
    '''
    storage = []
    directory_path = '../../data/2020/'
    files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    for i in range(len(files_list)):
        data_path = '../../data/2020/' + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            df = filtered_signal_ma(data_path, action, ma)
            returns = df.groupby(["year","stock"]).sum()["percent_return"]
            time_interval = df.groupby(["year","stock"]).sum()["holding_time_intervals"]
            metric = (returns/time_interval) * 252 * 13
            storage.append(metric)

    m = pd.concat(storage).groupby('year')
    return m.mean()

def backtest_ma(ma_list, action):
    '''
    Get annual return for each ma window

    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    ma_list : list
        A list of number indicates the size of moving average window

    Returns
    -------
    pandas.core.series.Series
        A series that contains all annualized return for each year of each window size.

    Example
    -------
    >>>get_annual('buy_only',3)    
    
    '''
    storage = []
    for ma in ma_list:
        storage.append(get_annual(action,ma))
    return storage
    df = pd.DataFrame(storage).T
    
def all_50(action,ma):
    '''
    Combine all filtered and processed
    stocks data.
    
    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    ma : int
        A integer indicates the size of MA window
    
    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe that contains all filtered stock data.
    
    Example
    -------
    >>>all_50('buy_only',5)
    '''
    directory_path = '../../data/2020/'
    files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    for i in range(len(files_list)):
        data_path = '../../data/2020/' + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:

            if i == 0:
                compiled_df_50 = filtered_signal_ma(data_path, action, ma)
            else:
                intermediate_df = filtered_signal_ma(data_path, action, ma)
                compiled_df_50 = pd.concat([compiled_df_50, intermediate_df])
    return compiled_df_50

def get_stats(action,ma):
    '''
    Get statistical measures of result, and summarize 
    in a data frame.
    
    
    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    ma : int
        A integer indicates the size of MA window.
    
    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe that contains all filtered stock data.
    
    Example
    -------
    >>>get_stats('buy_only',5)
    '''
    result = pd.DataFrame()
    df_all = all_50(action, ma)
    g = df_all.groupby('stock')

    df_lastrow = (pd.concat([g.tail(1)])
       .drop_duplicates()
       .sort_values('stock')
       .reset_index(drop=True))
    
    result['index'] = [ma]
    result['avg_metric'] = [get_annual(action,ma).mean()]
    result['std_metric'] = [get_annual(action,ma).std()]
    result['avg_metric/std_metric'] = result['avg_metric']/result['std_metric']
    result['filter_counts'] = [df_all.shape[0]]
    result['avg_port_cum_return'] = [df_lastrow['cumprod_portfolio_change'].mean()]
    result['avg_holding_time_per_signal'] = [df_all['holding_time_in_days'].mean()]
    return result