"""
This makes call to the quandl/yahoo database to get the data
@author Bala Bathula
"""

import os
import quandl
import datetime as dt
import pandas as pd
from requests_oauthlib import OAuth1
import simplejson as json
from ediblepickle import checkpoint

'''
with open("secrets/quandl_secrets.json.nogit") as fh:
    secrets = json.loads(fh.read())

print secrets
# create an auth object
auth = OAuth1(
    secrets["api_key"]
    
)
'''


cache_dir='cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

#@checkpoint(key=lambda args, kwargs:'_'.join(map(str, ['-'.join(args[0]),'-'.join(kwargs['features']),
#                                                       kwargs['start_dt'],kwargs['end_dt'],'.p'])), work_dir=cache_dir)
#@checkpoint(key=lambda args, kwargs:'_'.join(map(str, ['-'.join(args[0]),'symbols','-'.join(args[1]),'features','.p'])), work_dir=cache_dir)

#@checkpoint(key=lambda args, kwargs:args[0]+'.p', work_dir=cache_dir)

def get_data_from_quandl(symbol,features=['Close','Adj. Close','Open','Adj. Open'],
                         start_dt=dt.datetime(2000,1,1),
                         end_dt=dt.datetime.today()):
    """
    Gets the required data for the given symbol from quandl and store it as the csv file
    Store the file name as : {SYMBOL.csv}. This has some problems
    """
    
    
    quandl.ApiConfig.api_key = "MKXxiRmCQyr6ysZ5Qd2x"
    # quandl.ApiConfig.api_key=auth
    # For SPY use Yahoo instead of WIKI, but it will only give (Open, High,close, adjusted close, volume)
    if symbol=='SPY':
        
        dataframe=quandl.get('YAHOO/INDEX_SPY',start_date=start_dt.strftime('%Y-%m-%d'),
                             end_date=end_dt.strftime('%Y-%m-%d')) # This will be a pandas dataframe
        # Rename Adjusted Close to Adj_Close...I think
        dataframe=dataframe.rename(columns={'Adjusted Close':'Adj. Close'})
        data_source='YAHOO'
        # Return the dataframe only with the entries requested and remove the ones that are not present in YAHOO database
        df=pd.DataFrame(index=dataframe.index) # Create an empty dataframe with indices being dates
        
        used_features=[]; # If the user selects the features that are not part of YAHOO data set
        for ftrs in features:
            try:
                df=df.join(dataframe[ftrs])
                used_features.append(ftrs)
            except KeyError:
                pass
                #removed_features.append(ftrs) # Returns the features to the plot that are not found in dataset
                
        #for rm in removed_features: features.remove(rm)
        
        if len(used_features)==0:
            data_dict={'error':'Symbol/features not found'}
            return data_dict
        else:
            data_dict={'data':df,'features':used_features,'src':data_source}
            return data_dict
    else:
        #
        # WIKI database has 13 columns (Open,High,Low,Close,Volume, Ex-Dividend, Split Ratio,  Adj. Open,  Adj. High,  Adj. Low,  Adj. Close, Adj. Volume)
        try:
            dataframe=quandl.get('WIKI/%s'%(symbol),start_date=start_dt.strftime('%Y-%m-%d'),
                                 end_date=end_dt.strftime('%Y-%m-%d')) # This will be a pandas dataframe
            
            features=list(features); # Creates a new copy
            
            # Return the dataframe only with the entries requested
            # Quandl data frame already indexes to date (no need for any other change)
            df=pd.DataFrame(index=dataframe.index) # Create an empty dataframe with indices being dates
            
            for ftrs in features:
                df=df.join(dataframe[ftrs])
            
            data_source='WIKI'
            data_dict={'data':df,'features':features,'src':data_source}
            # return df,features,data_source,error_status
            
            return data_dict
        except quandl.errors.quandl_error.NotFoundError:
            data_dict={'error':'Symbol not found'}
            return data_dict

def get_data_from_quandl_old(symbol, base_dir='../data'):
    """
    Gets the required data for the given symbol from quandl and store it as the csv file
    Store the file name as : {SYMBOL.csv}. This has some problems
    """
    
    quandl.ApiConfig.api_key = "MKXxiRmCQyr6ysZ5Qd2x"
    if symbol=='SPY':
        dataframe=quandl.get('YAHOO/INDEX_SPY') # This will be a pandas dataframe
        # Here I need to rename Adjusted Close to Adj_Close...I think
    else:
        dataframe=quandl.get('EOD/%s'%(symbol)) # This will be a pandas dataframe
        
    dataframe.to_csv(os.path.join(base_dir,symbol+'.csv')) # Writes the data to the file
    return

def get_data_from_yahoo1(symbol,base_dir='../data'):
    """
    Directly query the website and load the content into a file
    """
    import requests
    base_url="http://ichart.finance.yahoo.com/table.csv?s="
    
    response = requests.get(base_url+symbol,stream=True)
    
    with open(os.path.join(base_dir,symbol+'.csv'),'wb') as data_file:
        data_file.write(response.content)
    
    return
    
def get_data_from_yahoo2(symbol,base_dir='../data'):        
    """
    This uses pandas_datareader
    Documentation: https://pandas-datareader.readthedocs.io/en/latest/
    """
    
    import pandas_datareader.data as pdr
    import datetime
    
    str_date=datetime.datetime(1950,1,1) # Start is kept very long ago so we can get data as much as we can
    end_date=datetime.datetime.today()
    
    df=pdr.get_data_yahoo(symbol,str_date,end_date); # This will be a dataframe
    
    df.to_csv(os.path.join(base_dir,symbol+'.csv')); # Write it into a file
    
    
    return

def get_data_from_yahoo3(symbol,base_dir='../data'):
    """
    This uses yahoo_finance python library
    """
    # TODO: 
    return
if __name__=='__main__':
    get_data_from_yahoo2('AAPL')