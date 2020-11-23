import numpy as np
import matplotlib.pyplot as plt

####################################################################

def bootstrap_sampling(input_array, n = 10000, dtype = "float", random_state = 1):
    """
    Performs bootstrap sampling of a given (1 x d) array.

    Parameters:
    -----------
    input_array : numpy.array
        Array of elements for bootstrap sampling.
    n : integer
        Number of bootstrap samples to take. Default value is 10000.
    dtype : string
        String input to determine type of outputs in bootstrap samples. Can be "float" (default) or "int".
    random_state : int
        Input for reproducible sampling. Default value is 1.
    
    Returns
    -------
    numpy.array
        Numpy array of dimensions (n x d)
    """
    # Setting random state for reproducible bootstrap sampling
    np.random.seed(random_state)
    
    bootstrap_samples_array = np.empty([len(input_array),n], dtype = dtype)

    for i in range(n):

        bootstrap_samples_array[:,i] = np.random.choice(input_array, size = len(input_array))
        
    return bootstrap_samples_array

####################################################################

def p_adjust_bh(p_array):
    """
    Performs Benjamini-Hochberg adjustment for p-values in multiple hypothesis testing.

    Parameters:
    -----------
    p_array : numpy.array
        (1 x d) array of unadjusted p values from multiple independent hypothesis testing.
    
    Returns
    -------
    numpy.array
        Numpy array of dimensions (1 x d)
    """
    boolean_descending_order = p_array.argsort()[::-1]
    boolean_original_order = boolean_descending_order.argsort()
    
    steps = float(len(p_array)) / np.arange(len(p_array), 0, -1)
    adjusted_pval = np.minimum(1, np.minimum.accumulate(steps * p_array[boolean_descending_order]))
    
    return adjusted_pval[boolean_original_order]

####################################################################

# Plotting function of bootstrap histograms for Buy Signals
def paired_histograms_buy(strategy_array, strategy_title, benchmark_buy_array):
    """
    Generate a visualisations of bootstrap sampling mean histograms for alternate & benchmark strategy. 
    Also generate a separate visualisation of one-tailed hypothesis test with the p-value.
    For Buy signals only.

    Parameters:
    -----------
    strategy_array : numpy.array
        (n x d) array of bootstrap samples for the alternate strategy.
    strategy_title : string
        String input for appropriate plot titles indicating alternate strategy name.
    benchmark_buy_array : numpy.array
        (n x d) array of bootstrap samples for the benchmark strategy.
    
    Returns
    -------
    Histogram plots with print statement on 1 tailed hypothesis test p-value.
    """
    fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    
    # AX 1: Bootstrap Sample Means Histograms
    ax1.hist(benchmark_buy_array.mean(axis = 0), bins = 50, density = True, color = "red", alpha = 0.5, label = "Benchmark: Filter(1+2)");
    ax1.hist(strategy_array.mean(axis = 0), bins = 50, density = True, color = "blue", alpha = 0.5, label = strategy_title);
    
    ax1.axvline(benchmark_buy_array.mean(axis = 0).mean(), color = "red")
    ax1.axvline(strategy_array.mean(axis = 0).mean(), color = "blue")
    ax1.set_title("Buy Signals Bootstrap Sampling of Filter(1+2) vs Best " + strategy_title)
    ax1.legend()
    
    # AX 2: Hypothesis Test (Alternate Strategy > Benchmark)
    ax2.hist((strategy_array - benchmark_buy_array).mean(axis = 0), bins = 50, density = True, color = "orange", alpha = 0.5, label = "Benchmark: Filter(1+2)");
    ax2.axvline(0, color = "black", linestyle = "--")
    ax2.set_title("Hypothesis Test: Mean of Best {} - Mean of Benchmark Filter(1+2) > 0".format(strategy_title))
    
    plt.show()
    
    pval = 1 - sum((strategy_array.mean(axis = 0) - benchmark_buy_array.mean(axis = 0)) >0)/len(strategy_array.mean(axis = 0))
    print("The p-value for hypothesis test (Mean of Best {0} - Mean of Benchmark Filter(1+2) > 0) is {1:.3f}".format(strategy_title, pval))

####################################################################

# Plotting function of bootstrap histograms for Sell Signals
def paired_histograms_sell(strategy_array, strategy_title, benchmark_sell_array):
    """
    Generate a visualisations of bootstrap sampling mean histograms for alternate & benchmark strategy. 
    Also generate a separate visualisation of one-tailed hypothesis test with the p-value.
    For Sell signals only.

    Parameters:
    -----------
    strategy_array : numpy.array
        (n x d) array of bootstrap samples for the alternate strategy.
    strategy_title : string
        String input for appropriate plot titles indicating alternate strategy name.
    benchmark_sell_array : numpy.array
        (n x d) array of bootstrap samples for the benchmark strategy.
    
    Returns
    -------
    Histogram plots with print statement on 1 tailed hypothesis test p-value.
    """
    fig,  (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
    
    # AX 1: Bootstrap Sample Means Histograms
    ax1.hist(benchmark_sell_array.mean(axis = 0), bins = 50, density = True, color = "red", alpha = 0.5, label = "Benchmark: Filter(1+2)");
    ax1.hist(strategy_array.mean(axis = 0), bins = 50, density = True, color = "blue", alpha = 0.5, label = strategy_title);
    
    ax1.axvline(benchmark_sell_array.mean(axis = 0).mean(), color = "red")
    ax1.axvline(strategy_array.mean(axis = 0).mean(), color = "blue")
    ax1.set_title("Sell Signals Bootstrap Sampling of Filter(1+2) vs Best " + strategy_title)
    ax1.legend()
    
    # AX 2: Hypothesis Test (Alternate Strategy > Benchmark)
    ax2.hist((strategy_array - benchmark_sell_array).mean(axis = 0), bins = 50, density = True, color = "orange", alpha = 0.5, label = "Benchmark: Filter(1+2)");
    ax2.axvline(0, color = "black", linestyle = "--")
    ax2.set_title("Hypothesis Test: Mean of Best {} - Mean of Benchmark Filter(1+2) > 0".format(strategy_title))
    
    plt.show()
    
    pval = 1 - sum((strategy_array.mean(axis = 0) - benchmark_sell_array.mean(axis = 0)) >0)/len(strategy_array.mean(axis = 0))
    print("The p-value for hypothesis test (Mean of Best {0} - Mean of Benchmark Filter(1+2) > 0) is {1:.3f}".format(strategy_title, pval))