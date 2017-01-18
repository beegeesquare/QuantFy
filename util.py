'''
These are some utility functions used for the first project in ML-Trading
@author Bala Bathula
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import apicall_data
from copy import deepcopy
import datetime as dt



def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    # Check if the data exists for the given symbol.  If not, then fetch it from the quandl database and save it to the base folder
    
    if os.path.isdir(base_dir)==False:
        #print ('making the directory')
        os.mkdir('data')
    
    if os.path.isfile(os.path.join(base_dir,"{}.csv".format(str(symbol)))):
        print('Data for symbol {0} exists'.format(str(symbol)));
    else:
        # Make an api call to the quandl database
        apicall_data.get_data_from_yahoo2(symbol, base_dir);
        print('Data copied to the local directory'.format(str(symbol)));
    
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


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

    #print df.head(10)
    
    return df
def plot_data(df, title="Stock prices",y_label='Price'):
    """Plot stock prices with a custom title and meaningful axis labels."""
    if os.path.isdir('plots')==False:
        os.mkdir('plots')
        
    plt.style.use('ggplot')
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    plt.show()
    plt.savefig('plots/price_vs_date_.png')


def get_rolling_mean(df, window):
    """Return rolling mean of given values, using specified window size."""
    # return pd.rolling_mean(values, window=window)
    rolling_df=pd.DataFrame(index=df.index)
    for sym in df.columns:
        df_tmp= pd.Series.rolling(df[sym],window=window).mean().to_frame() # to_frame converts series to a data frame
        rolling_df=rolling_df.join(df_tmp)
        
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
    return  rolling_df

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
   
    upper_band=rm+2*rstd
    lower_band=rm-2*rstd
    return upper_band, lower_band

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


def sharpe_ratio(adr,sddr,sf=252,rfr=0.0,):
    """ Computes the sharpe ratio"""
    return sf**(1.0/2)*(adr-rfr)/sddr


def requested_params():
    return

if __name__=='__main__':
    symbols=['SPY','GOOG']
    full_data=[(sym, apicall_data.get_data_from_quandl(sym, start_dt=dt.datetime.today()-dt.timedelta(days=5*365),
                                                       end_dt=dt.datetime.today())
                        ) for sym in symbols]
    
    df=get_data(full_data)   
    rm=get_rolling_mean(df,window=20)
    rstd=get_rolling_std(df, window=20)
    u_band,l_band=get_bollinger_bands(rm,rstd)
    #print u_band.columns
    #print l_band.columns