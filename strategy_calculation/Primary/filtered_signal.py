import pandas as pd
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join
pd.options.mode.chained_assignment = None
def filtered_signals_filter3(trade_signals, data_path,filter_type):
    """ 
    Read raw csv data, and split date time. Extract all trade signals 
    from oscillator, and apply wanted filter to filter out signals. Return
    a table containing information about profitability of each trade, as well
    as holding time.

    Parameters
    ----------
    trade_signals : str
        A string indicate whether we want to look at sell/buy signal only.
    data_path: str
        A string indicate the data user want to read.
    filter_type : int
        A integer indicates the wanted filter type. 0 -> no filter;
        1 -> polarity filter(positive/negtive);
        2 -> polarity filter + (absolute value > 7)
    
    Returns
    -------
    pandas.core.frame.DataFrame 
        A dataframe that contains filtered signal and corresponding information.
        
    Example
    --------
    >>> data_path = '../../data/2020/''
    >>> filtered_signals_filter3('buy_only',data_path,1)
    """
    df = pd.read_csv(data_path, delimiter=r"\s+",header=None)
    df.columns = ["date", "price", "oscillator"]
    df["date"] = df["date"].apply(lambda x: datetime.strptime(str(x), "1%y%m%d.00%H%M.00"))
    df = df.reset_index()
    df = df.rename(columns = {"index":"interval_index"})
    
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
    
    # Convert holding time to days float type
    df_signal["holding_time_in_days"] = df_signal["holding_time"]/pd.Timedelta(1.0,unit='D')
    
    # Remove unnecessary columns
    df_signal = df_signal.drop(["exit_date", "holding_time", "oscillator_lag1", 
                                "oscillator_lag2", "action_lag1", "action_joint"], axis = 1)

    # Drop last signal because it does not have an exit trade signal
    df_signal = df_signal[:-1]
    
    # Filter 1 & 2: Appropriate osc polarity & no trading if abs(osc) > 7 & Momentum Filter
    if filter_type == 2:
        filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) &\
                (abs(df_signal["oscillator"]) < 7) 
    if filter_type == 1:
        filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0)
    if filter_type == 0:
        filter_boolean = pd.Series([True]*df_signal.shape[0])       
    filtered_signal = df_signal[filter_boolean]
    
    if trade_signals == "buy_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==1]
    # Sell signals only
    elif trade_signals == "sell_only":
        filtered_signal = filtered_signal[filtered_signal["action"]==-1]
    # Both signals
    else:
        filtered_signal = filtered_signal
        
    # By year
    filtered_signal["year"], filtered_signal["month"] = filtered_signal["date"].dt.year,filtered_signal["date"].dt.month
    filtered_signal["cumprod_portfolio_change"] = filtered_signal["portfolio_change"].cumprod()
    filtered_signal['stock']=data_path[(len(data_path)-16):(len(data_path)-11)]
    return filtered_signal.reset_index(drop=True)


def get_annual(action,filter_type):
    '''
    Combine all filtered and processed
    stocks data, and calculate the annulized
    return for each year.

    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    filter_type : int
        A integer indicates the wanted filter type. 0 -> no filter;
        1 -> polarity filter(positive/negtive);
        2 -> polarity filter + (absolute value > 7)

    Returns
    -------
    pandas.core.series.Series
        A series that contains all annualized return for each year.

    Example
    -------
    >>>get_annual('buy_only',0)
            
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
            df = filtered_signals_filter3(action, data_path, filter_type)
            returns = df.groupby(["year","stock"]).sum()["percent_return"]
            time_interval = df.groupby(["year","stock"]).sum()["holding_time_intervals"]
            metric = (returns/time_interval) * 252 * 13
            storage.append(metric)

    m = pd.concat(storage).groupby('year')
    return m.mean()
    
def all_50(action,filter_type):
    '''
    Combine all filtered and processed
    stocks data.
    
    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    filter_type : int
        A integer indicates the wanted filter type. 0 -> no filter;
        1 -> polarity filter(positive/negtive);
        2 -> polarity filter + (absolute value > 7)
    
    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe that contains all filtered stock data.
    
    Example
    -------
    >>>all_50('buy_only',0)
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
                compiled_df_50 = filtered_signals_filter3(action, data_path,filter_type)
            else:
                intermediate_df = filtered_signals_filter3(action, data_path,filter_type)
                compiled_df_50 = pd.concat([compiled_df_50, intermediate_df])
    return compiled_df_50

def get_stats(action,filter_type):
    '''
    Get statistical measures of result, and summarize 
    in a data frame.
    
    
    Parameters
    ----------
    action : str
        A string indicate whether we want to look at sell/buy signal only.
    filter_type : int
        A integer indicates the wanted filter type. 0 -> no filter;
        1 -> polarity filter(positive/negtive);
        2 -> polarity filter + (absolute value > 7)
    
    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe that contains all filtered stock data.
    
    Example
    -------
    >>>get_stats('buy_only',0)
    '''
    result = pd.DataFrame()
    df_all = all_50(action, filter_type)
    g = df_all.groupby('stock')

    df_lastrow = (pd.concat([g.tail(1)])
       .drop_duplicates()
       .sort_values('stock')
       .reset_index(drop=True))
    
    result['index'] = [filter_type]
    result['avg_metric'] = [get_annual(action,filter_type).mean()]
    result['std_metric'] = [get_annual(action,filter_type).std()]
    result['avg_metric/std_metric'] = result['avg_metric']/result['std_metric']
    result['filter_counts'] = [df_all.shape[0]]
    result['avg_port_cum_return'] = [df_lastrow['cumprod_portfolio_change'].mean()]
    result['avg_holding_time_per_signal'] = [df_all['holding_time_in_days'].mean()]
    return result

