'''
These are some utility functions used for the first project in ML-Trading
@author Bala Bathula
'''

import os
import pandas as pd
import apicall_data
from copy import deepcopy
import datetime as dt



def get_data(list_data_tuples):
    """Read stock data (adjusted close) for given symbols """
    
       
    benchmark_symbol=list_data_tuples[0][0]; # First element is the benchmark symbol
    
    #print benchmark_symbol
    
    df=pd.DataFrame(index=list_data_tuples[0][1]['data'].index) # First dataframe index is nothing but date
    
    for tpl in list_data_tuples:
        #print tpl[0]
        df_temp = pd.DataFrame(tpl[1]['data']['Adj. Close'],index=tpl[1]['data'].index)
        df_temp = df_temp.rename(columns={'Adj. Close': tpl[0]}) # tpl[0] is the symbol
        #print df_temp,tpl[0]
        df = df.join(df_temp)
        if tpl[0] == benchmark_symbol:  # drop dates SPY did not trade
            df = df.dropna(subset=[benchmark_symbol])

    df=df.dropna(axis=0) # This drops any NaN values especially if the stock price has no information
    
    return df


def get_rolling_mean(df, window):
    """Return rolling mean of given values, using specified window size."""
    # return pd.rolling_mean(values, window=window)
    rolling_df=pd.DataFrame(index=df.index)
    for sym in df.columns:
        df_tmp= pd.Series.rolling(df[sym],window=window).mean().to_frame() # to_frame converts series to a data frame
        rolling_df=rolling_df.join(df_tmp)
    
    # Update the NaN values which are initial values for the 
    
    rolling_df.ix[:window, :] = 0
    
    return rolling_df
    #return pd.Series.rolling(values,window=window).mean()


def get_rolling_std(df, window):
    """Return rolling standard deviation of given values, using specified window size."""
    rolling_df=pd.DataFrame(index=df.index)
    
    for sym in df.columns:
        df_tmp= pd.Series.rolling(df[sym],window=window).std().to_frame() # to_frame converts series to a data frame
        rolling_df=rolling_df.join(df_tmp)
    
    # return pd.rolling_std(values,window=window) # Older version of pandas
    # return pd.Series.rolling(values,window=window).std()
    
    # Update the NaN values which are initial values for the 
    
    
    rolling_df.ix[:window, :] = 0
    
    return  rolling_df

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
   
    upper_band=rm+2*rstd
    lower_band=rm-2*rstd
    return upper_band, lower_band

def compute_normalized_BB(df,rm,rstd,window):
    """
    Return the bollinger bands feature (which is the normalized values) so that it will typically provide values between -1.0 and 1.0
    
    """
    normalized_bb=pd.DataFrame(index=df.index)
    
    for sym in df.columns:
        df_tmp= (df[sym] - rm[sym])/(2 * rstd[sym])
        normalized_bb=normalized_bb.join(df_tmp)
    
    # Update the NaN values which are initial values for the 
    
    normalized_bb.ix[:window, :] = 0
    
    return normalized_bb
    
def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    # Initialize the dataframe here
    daily_returns=df.copy(); # Make a copy of original dataframe
    daily_returns[1:]=(df[1:]/df[:-1].values)-1 # .values is very important here. Otherwise pandas does the arithmetic operation based on index rather than shifted values
    
    if (df.size==df.shape[0]): # That means there is only one column
        daily_returns.ix[0]=0
    else:
        # Make the value at index 0 to 0 for all columns
        daily_returns.ix[0,:]=0; # 
    # Other way to do this is following
    # daily_returns=(df/df.shift(1))-1; # However the zero index will NaN
    # daily_returns.ix[0,:]=0
    
    
    return daily_returns

def cummulative_return(df):
    """
    Computes the cummulative return for the df
    (portfolio_value[-1]/portfolio_value[0])-1
    """
    cumm_daily_return=(df[-1]/df[0])-1
    
    return cumm_daily_return


def sharpe_ratio(adr,sddr,sf=252,rfr=0.0):
    """ Computes the sharpe ratio"""
    rfr=((1.0 + rfr) ** (1/sf)) - 1 # Daily risk free return. This is the shortcut to calculate daily (sf=252) risk free return
    return sf**(1.0/2)*(adr-rfr)/sddr

def compute_momentum(df,window):
    '''
    Computes the momentum of the stock, which is computed as:
    momentum[t] = (price[t]/price[t-N]) - 1
    '''
    # Initialize the momentum dataframe with df
    momentum=df.copy()
    momentum[window:] = (df[window:]/df[:-window].values) - 1
    momentum.ix[:window, :] = 0
    
    return momentum

def compute_volatility(df,window):
    '''
    Computes the standard-deviation of the daily return
    '''
    
    daily_returns=compute_daily_returns(df)
    
    volatility=get_rolling_std(daily_returns,window)
    
    # Put the initial values to zeros,
    volatility.ix[:window, :] = 0
    
    return volatility


def requested_params():
    return

if __name__=='__main__':
    symbols=['SPY','GOOG']
    full_data=[(sym, apicall_data.get_data_from_quandl(sym, start_dt=dt.datetime.today()-dt.timedelta(days=5*365),
                                                       end_dt=dt.datetime.today())
                        ) for sym in symbols]
    print full_data
    df=get_data(full_data)   
    rm=get_rolling_mean(df,window=20)
    rstd=get_rolling_std(df, window=20)
    u_band,l_band=get_bollinger_bands(rm,rstd)
    #print u_band.columns
    #print l_band.columns