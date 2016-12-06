"""
This makes call to the quandl database to get the data
@author Bala Bathula
"""

import os


def get_data_from_quandl(symbol, base_dir='../data'):
    """
    Gets the required data for the given symbol from quandl and store it as the csv file
    Store the file name as : {SYMBOL.csv}. This has some problems
    """
    import quandl
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