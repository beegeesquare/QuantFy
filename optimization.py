"""
This project allows to find how much of portfolio funds should be allocated to each stock as to optimize it's performance.
Objective function is to maximize sharpe ratio
@author Bala Bathula
"""
import datetime 
import util
import numpy as np
import pandas as pd
import scipy.optimize as spo

def sharpe_function(allocs,df,sv=1000000,sf=252.0,rfr=0.0):
    """Takes the allocations and computes the sharpe ratio"""
   
    # 1. Normalize the prices according to the first day
    df=df/df.ix[0,:]
    
    # 2. Multiply each column by allocation to the corresponding equity 
    df=allocs*df # Number of columns in df should be same as the elements in alloc
    
    # 3. Multiply these normalized allocations by starting value of overall portfolio, to get position values
    df=sv*df; # sv is the start-value
    
    # 4. Sum each row (i.e., all position values for each day). That is your portfolio value
    df=df.sum(axis=1) # This gives daily portfolio value
    
    dr_df=util.compute_daily_returns(df)
    adr=dr_df.mean(); # This is daily-return data frame
    
    # 5c. Standard deviation
    sddr=np.std(dr_df)
    
    # 5d. Sharpe ratio
    sr= util.sharpe_ratio(adr=adr,sddr=sddr,sf=sf,rfr=0.0)
    
    # -1 is multiplied as max=min*-1
    
    return sr*-1

def access_portfolio(df,bench_sym,allocs,sv,rfr=0.0,sf=252.0):
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
   
    
    # If SPY exists in the df, then drop it (we need SPY to get the dates for all the trading dates)
    normalized_plot_df=pd.DataFrame(df[bench_sym]/df[bench_sym][0],index=df.index)
    
    df=df.drop(bench_sym,axis=1)
    
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
    
    
    return cr,adr,sddr,sr,ev,normalized_plot_df


def optimize_portfolio(df,bench_sym,sv):
    
    """
    Outputs:
    
    allocs: A 1-d Numpy ndarray of allocations to the stocks. All the allocations must be between 0.0 and 1.0 and they must sum to 1.0.
    cr: Cumulative return
    adr: Average daily return
    sddr: Standard deviation of daily return
    sr: Sharpe ratio (risk adjustment ratio)

    The input parameters are:

    sd: A datetime object that represents the start date
    ed: A datetime object that represents the end date
    syms: A list of symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    gen_plot: If True, create a plot named plot.png
    """
    
        
    # If SPY exists in the df, then drop it (we need SPY to get the dates for all the trading dates)
    # normalized_plot_df=pd.DataFrame(df.SPY/df.SPY[0],index=df.index)
    
    full_data=pd.DataFrame(df,index=df.index)
    
    df=df.drop(bench_sym,axis=1)
    
    sym=df.columns; # This is after dropping the SPY
    
    # Here we need to maximize sharpe ratio by fitting the allocs values
    
    initial_alloc=[(1.0/len(sym))]*len(sym) # *len(sym) will create a len(sym) list
    
    # Each alloc is bounded by (0,1) =(<low>,<high>) and there are len(syms) such variables
    bounds=((0,1),)*len(sym) # , creates nested tuples, where each tuple is (0,1)
    
    # Run the minimization function
    # Constraints should be passed as sequence of dict, i.e, each element of the tuple is a dictionary. Say if we have 5 constraints, then
    # Each will be an element of tuple, so there are 5 tuples.
    
    opt_allocs=spo.minimize(sharpe_function, initial_alloc, args=df, method='SLSQP', bounds=bounds, 
                            constraints=({'type':'eq','fun':lambda opt_allocs: 1-np.sum(np.abs(opt_allocs))})); # Constraints that the sum of opt_allocs should be one
    
    optimal_allocs=opt_allocs.x # Gives the optimal solution
    
    
    
    # Now given these optimal alloc compute the portfolio
    # Send the full_data here it should also include the bench_sym
    cr,adr,sddr,sr,ev,normalized_plot_df=access_portfolio(full_data,bench_sym,optimal_allocs,sv)
    
    
    
    return cr, adr, sddr,sr,ev, normalized_plot_df, optimal_allocs



if __name__=='__main__':
    print('Optimize portfolio')
    sd=datetime.datetime(2010,01,01)
    ed=datetime.datetime(2010,12,31)
    
    '''
    # symbols=['GOOG', 'AAPL', 'GLD', 'XOM'];
    symbols=['AXP', 'HPQ', 'IBM', 'HNZ']
    cr,adr,sddr,sr,optimal_allocs=optimize_portfolio(sd=sd,ed=ed,sym=symbols,gen_plot=True)
    
    print('Start date: {0}'.format(sd.strftime('%Y-%m-%d')))
    print('End date: {0}'.format(ed.strftime('%Y-%m-%d')))
    print('Symbols: {0}'.format(symbols))
    print('Allocatons: {0}'.format(optimal_allocs))
    # Outputs
    print('Sharpe ratio: {0}'.format(sr))
    print('Volatility (std of daily returns): {0}'.format(sddr))
    print('Average daily return: {0}'.format(adr))
    print('Cummlative return: {0}'.format(cr))
    '''
    