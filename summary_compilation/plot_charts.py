import matplotlib.pyplot as plt

def plots(data, signal_type='buy', stat='return'):
    """
    Generate specific plots of descriptive statistics for either Buy or Sell signals across the best of all strategy types.

    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame object containing compiled descriptive statistics for all strategy types.
    signal_type : string
        String input stating the type of trade signals. Used for appropriate plot title.
    stat : string
        String input for appropriate plot titles indicating alternate strategy name.

    Returns
    -------
    Single plot of specified descriptive statistic across the best of all strategy types.
    """

    # Specify X-Axis
    x_axis = data.avg_metric.index

    if stat == 'return':
        # Average ANNUALISED Return Per Strategy
        plt.figure(figsize=(15,5))
        clrs = ['blue' if (x < max(data.avg_metric)) else 'green' for x in data.avg_metric]
        plt.bar(x_axis, data.avg_metric.values*100, color=clrs)
        plt.axhline(0, c='r')
        plt.xticks(rotation=45)
        plt.title(signal_type.upper() + ": Average Annualised Return Per Strategy")
        plt.ylabel("% Return")
        plt.xlabel(' ')
        plt.tight_layout()
        plt.show()
    if stat == 'return_std':
        # Average ANNUALISED Return Per Standard Deviation
        plt.figure(figsize=(15,5))
        clrs = ['blue' if (x < max(data['avg_metric/std_metric'])) else 'green' for x in data['avg_metric/std_metric']]
        plt.bar(x_axis, data['avg_metric/std_metric'].values, color=clrs)
        plt.axhline(0, c='r')
        plt.xticks(rotation=45)
        plt.title(signal_type.upper() + ": Average Annualised Return/Standard Deviation Per Strategy")
        plt.ylabel("Reward-Risk Ratio")
        plt.xlabel(' ')
        plt.tight_layout()
        plt.show()
    if stat == 'holding_time':
        # Average Holding Time
        plt.figure(figsize=(15,5))
        plt.bar(x_axis, data['avg_holding_time_per_signal'].values)
        plt.axhline(0, c='r')
        plt.xticks(rotation=45)
        plt.title(signal_type.upper() + ": Average Signal Holding Time Per Strategy")
        plt.ylabel("Holding Time (Days)")
        plt.xlabel(' ')
        plt.tight_layout()
        plt.show()

    if stat == 'counts':
        # Filter Counts Per Strategy
        plt.figure(figsize=(15,5))
        plt.bar(x_axis, data['filter_counts'].values)
        plt.axhline(0, c='r')
        plt.title(signal_type.upper() + ": Filter Counts Per Strategy")
        plt.xticks(rotation=45)
        plt.ylabel("Trades")
        plt.xlabel(' ')
        plt.tight_layout()
        plt.show()

    if stat == 'cumulative':
    # Average Portfolio Cumulative Return Per Strategy
        plt.figure(figsize=(15,5))
        clrs = ['blue' if (x < max(data['avg_port_cum_return'])) else 'green' for x in data['avg_port_cum_return']]
        plt.bar(x_axis, data['avg_port_cum_return'].values, color=clrs)
        plt.axhline(0, c='r')
        plt.xticks(rotation=45)
        plt.title(signal_type.upper() + ": Average Portfolio Cumulative Return Per Strategy")
        plt.ylabel("Average Cumulative Portfolio Return")
        plt.xlabel(' ')
        plt.tight_layout()
        plt.show()
