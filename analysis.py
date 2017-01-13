'''
This is the project to analyizes portfolio
A portfolio is a collection of stocks (or other assets) and corresponding allocations of funds to each of them. 
In order to evaluate and compare different portfolios, we first need to compute certain metrics, based on available historical data. 
@author Bala Bathula
'''
import datetime 
import pandas as pd
import util
import numpy as np





def access_portfolio(sd=datetime.datetime(2008,1,1),ed=datetime.datetime(2009,1,1), sym=['GOOG','AAPL','GLD','XOM'],
                     allocs=[0.1,0.2,0.3,0.4],sv=1000000,rfr=0.0,sf=252.0,gen_plot=False):
    """
    This function computes statistics for the portfolio
    Usage:
    cr, adr, sddr, sr, ev = assess_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), syms=['GOOG','AAPL','GLD','XOM'],allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, gen_plot=False)
    Outputs:
    cr: Cumulative return
    adr: Average period return (if sf == 252 this is daily return)
    sddr: Standard deviation of daily return
    sr: Sharpe ratio
    ev: End value of portfolio
    
    Inputs:
    
    sd: A datetime object that represents the start date
    ed: A datetime object that represents the end date
    syms: A list of symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    allocs: A list of allocations to the stocks, must sum to 1.0
    sv: Start value of the portfolio
    rfr: The risk free return per sample period for the entire date range (a single number, not an array).
    sf: Sampling frequency per year
    gen_plot: If True, create a plot named plot.png
    
    """
    # First get data for the date and symbols
    dt_range=pd.date_range(sd, ed) # Create series of dates for which the data needs to be obtained
    
    df=util.get_data(sym, dt_range); # This gets the adjust close value of the stock for each date and date frame index is date
    
    # If SPY exists in the df, then drop it (we need SPY to get the dates for all the trading dates)
    normalized_plot_df=pd.DataFrame(df.SPY/df.SPY[0],index=df.index)
    
    df=df.drop('SPY',axis=1)
    
    # To compute daily portfolio, we need to follow these steps:
    # 1. Normalize the prices according to the first day
    df=df/df.ix[0,:]
    
    # 2. Multiply each column by allocation to the corresponding equity 
    df=allocs*df # Number of columns in df should be same as the elements in alloc
    
    # 3. Multiply these normalized allocations by starting value of overall portfolio, to get position values
    df=sv*df; # sv is the start-value
    
    # 4. Sum each row (i.e., all position values for each day). That is your portfolio value
    df=df.sum(axis=1) # This gives daily portfolio value
    
    # 5. Compute statistics from the total portfolio value
    
    # 5a. Cummulative daily return
    cr= util.cummulative_return(df);
    
    # 5b. Average daily return
    dr_df=util.compute_daily_returns(df)
    adr=dr_df.mean(); # This is daily-return data frame
    
    # 5c. Standard deviation
    sddr=np.std(dr_df)
    
    # 5d. Sharpe ratio
    sr=util.sharpe_ratio(adr=adr,sddr=sddr,sf=sf,rfr=rfr)
    
    # sr=sf**(1.0/2)*(adr-rfr)/sddr
    
    # 5e. End value of the portfolio (How much your portfolio values at the end of the duration)
    
    ev=df.ix[-1];
    
    # Plot the normalized portfolio again normalized SPY
    normalized_plot_df=normalized_plot_df.join(pd.DataFrame(df/df.ix[0],columns=['Portfolio']),how='inner')
    if gen_plot==True:
        util.plot_data(normalized_plot_df, title='Daily portfolio value and SPY',y_label='Normalized price')
    
    return cr,adr,sddr,sr,ev

if __name__=='__main__':
    # Example 1:
    # cr,adr,sddr,sr,ev=access_portfolio(sd=datetime.datetime(2010,01,01),ed=datetime.datetime(2010,12,31),allocs=[0.2,0.3,0.4,0.1])
    
    sd=datetime.datetime(2010,01,01)
    ed=datetime.datetime(2010,12,31)
    allocs=[0.1, 0.2, 0.3, 0.4]
    symbols=['AXP', 'HPQ', 'IBM', 'HNZ'];
    
        
    cr,adr,sddr,sr,ev=access_portfolio(sd=sd,ed=ed,allocs=allocs,sym=symbols,gen_plot=True)
    
    print('Start date: {0}'.format(sd.strftime('%Y-%m-%d')))
    print('End date: {0}'.format(ed.strftime('%Y-%m-%d')))
    print('Symbols: {0}'.format(symbols))
    print('Allocatons: {0}'.format(allocs))
    # Outputs
    print('Sharpe ratio: {0}'.format(sr))
    print('Volatility (std of daily returns): {0}'.format(sddr))
    print('Average daily return: {0}'.format(adr))
    print('Cummlative return: {0}'.format(cr))
    
    
    
    