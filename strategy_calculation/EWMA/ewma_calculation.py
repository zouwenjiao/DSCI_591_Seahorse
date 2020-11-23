import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join
from tqdm.notebook import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

directory_path = "../../data/2020/"
files_list = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

def filtered_signals_ewma_slope_threshold(data_path, trade_signals, ewma_window = 5):
    """
    This function is used to backtest the EWMA strategy for a specific stock.

    Parameters:
    -----------
    data_path (str):
        The file-path for the data files
    trade_signals (str):
        Choose the strategy from the three selections: "both", "buy_only", "sell_only"
    ewma_window (int):
        The window size parameter for the EWMA indicator

    Output:
    -------
    pandas DataFrame:
        output of key backtest statistics
    """
    ##############################################################
    # Reading in data
    ##############################################################
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
                                       # no min period
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
    df_signal["holding_time_in_days"] = df_signal["holding_time"]/pd.Timedelta(1.0,unit='D')
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
    filter_boolean = (df_signal["oscillator"] * df_signal["action"] < 0) & \
    (abs(df_signal["oscillator"]) < 7) & (df_signal["ewma_price_gradient_percent_action"] >0)

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
    elif trade_signals == "both":
        filtered_signal= filtered_signal
    else:
        raise ValueError("Please ensure you pick a proper signal: buy_only, sell_only or both")

    #print("Number of '{0}' filtered signals = {1}".format(trade_signals.upper(), len(filtered_signal)))

    ##############################################################
    # Cumprod for dataset
    ##############################################################
    # Calculate cumprod for whole dataset
    filtered_signal["cumprod_portfolio_change"] = filtered_signal["portfolio_change"].cumprod()

    ##############################################################
    # Extract year and month for aggregation basis
    ##############################################################
    # Extract year and month from datetime for aggregation calculation
    filtered_signal["year"], filtered_signal["month"] = filtered_signal["date"].dt.year, filtered_signal["date"].dt.month

    # Creating stock ticker by doing chained splitting
    filtered_signal['stock']= str(data_path).split("/")[-1].split("2")[0]

    return filtered_signal

def optimizer(ewma_window, trade_signals):
    """
    This function uses the output from the `filtered_signals_ewma_slope_threshold`
    function and backtests the results across all the stocks to get the return statistics.

    Parameters:
    ----------
    ewma_window (int):
        The moving average window parameter
    trade_signals (str):
        Choose the strategy from the three selections: "both", "buy_only", "sell_only"

    Output:
    ------
    annualised_df (pandas DataFrame):
        A dataframe of 20 years of "annualised" backtested returns
    stat_df (pandas DataFrame):
        A dataframe of the key statistics (ie: filter counts, average
        cumulative return, average holding time)
    """
    annualised_storage = []
    stat_storage = []

    for i in tqdm(range(len(files_list))):
        data_path = "../../data/2020/" + files_list[i]

        # Failed data with no white space separator
        if files_list[i] in ["algn20years.txt","wynn20years.txt","klacl20years.txt","lvs20years.txt"]:
            pass

        else:
            z = filtered_signals_ewma_slope_threshold(data_path, trade_signals, ewma_window)

            # Part 1 - Store Annualised Returns
            returns = z.groupby(["year"]).sum()["percent_return"]
            time_interval = z.groupby(["year"]).sum()["holding_time_intervals"]
            metric = (returns/time_interval) * 252 * 13
            annualised_storage.append(metric)

            # Part 2 - Store the Summary Statistics
            # Number of trades
            trades = z.shape[0]
            # Calculate cumulative value of investing $1 during investing period
            cum_return_value = z.cumprod_portfolio_change.iloc[-1]
            # Holding time days
            holding_time_in_days = z.holding_time_in_days.sum()
            stat_storage.append([trades, cum_return_value, holding_time_in_days])

    # Convert the storage list into a dataframe and calculate the appropriate statistics
    statistics = pd.DataFrame(stat_storage)
    # Sum of all trades
    total_trades = statistics.sum()[0]
    # Avg of all compounded returns
    avg_cum_ret = statistics.mean()[1]
    # Sum of holding time for trades
    hold_time_sum = statistics.sum()[2]
    # Calculate average holding time per trade
    avg_holding_time = hold_time_sum/total_trades


    # Wrap up data into two separate dataframes
    stat_df = pd.DataFrame([total_trades, avg_cum_ret, avg_holding_time]).T
    stat_df.columns = ['filter_counts', 'avg_port_cum_return', 'avg_holding_time_per_signal']

    annualised = pd.DataFrame(annualised_storage).T
    annualised_df = annualised.mean(axis=1)

    return annualised_df, stat_df

def backtester(ewma_list, trade_signals, save_csv=True):
    """
    This function uses the `optimizer` function and iterates over a range of EWMA window
    parameter values to calculate the backtest statistics for each respective strategy.

    Parameters:
    ----------
    ewma_list (list):
        The range of integers to search over for the EWMA window parameter
    trade_signals (str):
        Choose the strategy from the three selections: "both", "buy_only", "sell_only"
    save_csv (bool):
        Set to `True` in order to save the output data as a csv file

    Output:
    ------
    pandas DataFrame:
        Outputs a dataframe with the key backtest statistics for the top strategy
    """
    # Step 1: Loop through the list of EWMA parameters and store each result
    storage = []
    for ewma in tqdm(ewma_list):
        storage.append(optimizer(ewma, trade_signals))

    # Step 2: Extract the annualised returns and statistics dataframe from step 1
    annualised_returns = pd.DataFrame([[i][0][0] for i in storage]).T
    statistics = pd.concat([i[1] for i in storage]).T
    annualised_returns.columns = ["EWMA" + "(" +str(i) + "/13)" for i in ewma_list]
    statistics.columns = ["EWMA" + "(" +str(i) + "/13)" for i in ewma_list]

    # Step 3: Calculate the appropriate statistics
    avg_annualised = annualised_returns.mean()
    std_annualised = annualised_returns.std()
    reward_risk_annualised = avg_annualised/std_annualised

    # Step 4: Combine all dataframes into one collective dataframe
    calculated_stats = pd.DataFrame([avg_annualised, std_annualised, reward_risk_annualised])
    extracted_statistics = pd.DataFrame(statistics.values)
    extracted_statistics.columns = calculated_stats.columns

    df = pd.concat([calculated_stats, extracted_statistics]).T
    col_name = ['avg_metric', 'std_metric','avg_metric/std_metric',
                'filter_counts', 'avg_port_cum_return', 'avg_holding_time_per_signal']
    df.columns = col_name
    if save_csv==True:
        df.to_csv("top_EWMA_"+str(round(time.time()))+"_"+str(trade_signals)+".csv")
    return df
