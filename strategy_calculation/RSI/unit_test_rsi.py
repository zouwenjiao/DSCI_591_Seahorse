import rsi_calculation as rsi
import pandas as pd 

# Test param
buy_rsi_window_list = [i+2 for i in range(2)]

data_directory_path = "../../data/2020/"

def unit_test_functions():
    print("Phase 1 unit test commencing:")
    buy_df = rsi.strategy_eval_rsi("buy_only", data_directory_path, buy_rsi_window_list)
    # Ensure the returned variable is a DataFrame
    assert isinstance(buy_df, pd.DataFrame)
    # Ensure the number of rows matches number of years (21)
    assert buy_df.shape[0] == 21
    # Ensure the number of strategies evaluated is correct
    assert buy_df.shape[1]== 2
    print("Phase 2 unit test commencing:")
    top_buy = rsi.get_stats_rsi(buy_df, "buy_only", data_directory_path)
    # Ensure the returned variable is a DataFrame
    assert isinstance(top_buy, pd.DataFrame)
    # Ensure that only the top strategy is returned
    assert top_buy.shape[0] == 1
    # Ensure the correct number of calculated descriptive statistics is covered
    assert top_buy.shape[1]== 6
if __name__ == "__main__":
    unit_test_functions()
    print("RSI unit tests passed.")
