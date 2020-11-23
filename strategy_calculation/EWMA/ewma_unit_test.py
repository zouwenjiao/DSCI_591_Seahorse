import ewma_calculation as em
import pandas as pd

def unit_test_functions():
    # Initialise the backtesting function
    range_search = [1,2]
    df = em.backtester(range_search, 'buy_only', save_csv=False)
    # Ensure the returned variable is a DataFrame
    assert isinstance(df, pd.DataFrame)
    # Ensure the number of rows matches the number of backtests
    assert df.shape[0]==len(range_search)
    # Ensure the number of columns is constant (6)
    assert df.shape[1]==6

if __name__ == "__main__":
    unit_test_functions()
    print("EWMA unit tests passed.")
