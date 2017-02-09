from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for
from werkzeug.utils import secure_filename
import os
from ediblepickle import checkpoint

cache_dir='cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
    UPLOAD_FOLDER = 'cache'
else: 
    UPLOAD_FOLDER = 'cache'


ALLOWED_EXTENSIONS = set(['csv'])
FILENAME=''; # This is a global variable

from collections import defaultdict

import numpy as np
from bokeh.plotting import figure, show,output_file,ColumnDataSource
from bokeh.palettes import viridis
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from bokeh.charts import Area



import pandas as pd
import datetime as dt
import util
import apicall_data

app_marketsim=Flask(__name__)
app_marketsim.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




marketSim = Blueprint('market_simulator.html', __name__, template_folder='templates')
@marketSim.route('/marketSimUpload',methods=['GET','POST'])
def marketsimUpload():
    if request.method=='GET':
        return render_template('market_simulator.html')
    else:
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url) # returns the same url
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url) # returns the same url
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # This will be with .csv extension
            # print filename
            
            file.save(os.path.join(app_marketsim.config['UPLOAD_FOLDER'], filename))
            # using the filename construct the dataframe
            uploadStatus='<font size="3" color="red" > File %s has been uploaded successfully </font>'%(filename)
            app_marketsim.config['UPLOAD_FILENAME']=filename
            return render_template('market_simulator.html',upload_success=uploadStatus)
            #return redirect(url_for('market_simulator.html.get_portVals',filename=filename))
        
        
        #return render_template('market_simulator.html')
@marketSim.route('/marketSim',methods=['GET','POST'])
def get_portVals():
    #if request.method=='GET':
    #    redirect(url_for('market_simulator.html.marketsimUpload'))
    if request.method=='POST':
        app_marketsim.vars={}
        filename=app_marketsim.config['UPLOAD_FILENAME']
        # print filename
        app_marketsim.vars['start_value']=float(request.form['start_value'])
        app_marketsim.vars['leverage_threshold']=float(request.form['leverage_threhsold'])
        app_marketsim.vars['bench_sym']=request.form['bench_sym']
        
        portvals,all_prices_df,param_df=compute_portvals(filename, start_val = app_marketsim.vars['start_value'],
                                  leverage_threshold= app_marketsim.vars['leverage_threshold'],
                                  bench_sym=app_marketsim.vars['bench_sym'])
                
        bench_df=pd.DataFrame(all_prices_df[app_marketsim.vars['bench_sym']],index=all_prices_df.index)
        
        # Now normalized portfolio graph against the benchmark symbol
        # Plot the comparison chart for the normalized data
        portval_performance=pd.DataFrame(bench_df/bench_df.ix[0]) # This is the performance df for the portfolio against Benchmark symbol
        portval_performance=portval_performance.join(portvals/portvals.ix[0])
        
        
        
        hover=HoverTool(
            tooltips=[
                ("Leverage",'@Leverage')
                ]
        )
                
        # print portval_performance
        source=ColumnDataSource(data=portval_performance)
        
        TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
        
        p = figure(width=1200, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
        #p=Area(adj_prices_df, width=1200, height=500,tools=[TOOLS,hover])
        p.title.text = "Portfolio comparison for the given order file %s"%(app_marketsim.config['UPLOAD_FILENAME'])
        p.legend.location = "top_left"
        
        colors=['red','blue']
        
        for (i,sym) in enumerate(portval_performance):
            p.line(portval_performance.index,portval_performance[sym],color=colors[i],line_width=2,legend=sym)
        
        #p.line('Date',app_marketsim.vars['bench_sym'],color=colors,line_width=2,source=source)
        
        script_port_comp, div_port_comp=components(p)
        
        return render_template('market_simulator.html',script_port_comp=script_port_comp, div_port_comp=div_port_comp,
                               computed_params=param_df.to_html())

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_orders(orders_file):
    
    # Create a dataframe for orders
   
    orders_df=pd.read_csv(orders_file,index_col='Date',parse_dates=True,na_values=['nan']) 
    # Sort the dataframe based on the date (which is index)    
    orders_df=orders_df.sort_index()
    return orders_df

def compute_portvals(orders_file, start_val = 1000000,leverage_threshold=2.0,bench_sym='SPY'): # Default file is orders.csv, but the test function calls other file
       
    
    #print os.path.join(UPLOAD_FOLDER,orders_file)   
    orders_df=get_orders(os.path.join(UPLOAD_FOLDER,orders_file)) # Pass the filename that was just uploaded
    #print orders_df
    
    
    list_symbols=list(orders_df.Symbol.unique()) # This gives the list of unique symbols in the order dataframe
    
    list_symbols.insert(0,bench_sym); # Insert the default symbol in the symbols list
    
    # Start date should the first date in the orders file   (and orders file is sorted based on dates)
    start_date = orders_df.index[0]
    # End date should be the last date entry in the orders file
    end_date = orders_df.index[-1]
    # get the price data for all the symbols in the order data frame for the date range in the order file
    # Here just get the data for the 'Adj. Close'
    
    full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=['Adj. Close'], start_dt=start_date,end_dt=end_date)
                        ) for sym in list_symbols]
    
    
    # Drop the bench mark price
    
    
    # Convert this to required format
    prices_df=util.get_data(full_data)
    
    all_prices_df=pd.DataFrame(prices_df)
    
    all_prices_df.to_csv('cache/all_prices.csv')
    
    prices_df=prices_df.drop(bench_sym,axis=1)
    
    # Add the 'Cash' key to the price dataframe with value =1.0
    prices_df['Cash']=1.0; # This add 1.0 to all rows in the prices_df
        
    # create a trades dataframe and initialize
    trades_df=pd.DataFrame(np.zeros((len(prices_df.index),len(list_symbols))),index=prices_df.index,columns=list_symbols)
    trades_df['Cash']=0;
    
    # create holding dataframe and initialize
    holding_df=pd.DataFrame(np.zeros((len(prices_df.index),len(list_symbols))),index=prices_df.index,columns=list_symbols)
    
    # Initialize  the cash in holding to the start value
    holding_df['Cash']=start_val; # Holding cash is same as the 'Cash' in trades_df, except we have start value initialized
    
    # negative value of stocks in holding_df could indicate that it has  
    
    # create values dataframe and initialize
    values_df=pd.DataFrame(np.zeros((len(prices_df.index),len(list_symbols))),index=prices_df.index,columns=list_symbols)
    
    values_df['value']=0;
    
    # create empty dataframe for leverage
    
        
    for (d,rows) in orders_df.iterrows(): # Here d is the date index # Order file assumes that if the date is same, then they are listed as twice
        if rows['Order']=='BUY':
            trades_df.loc[d,rows['Symbol']]=rows['Shares']
            trades_df.loc[d,'Cash']+=-trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']]; # -ve indicates we have purchased the stock
            # holding_df 'Cash' is the same as the trading except we have initialized to start_val
            holding_df.loc[d:,'Cash']+=-trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']];
            
            
        elif rows['Order']=='SELL': # Negative value indicates you are selling
            trades_df.loc[d,rows['Symbol']]=-rows['Shares']
            trades_df.loc[d,'Cash']+=-trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']]; # This will be +ve since we are selling the stock and it adds to the cash value
            
            holding_df.loc[d:,'Cash']+=-trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']];
            
        
        holding_df.loc[d:,rows['Symbol']]+=trades_df.loc[d,rows['Symbol']] 
        # After we update the number of holdings for each symbol and assuming only new orders are affecting the leverage, compute the leverage for this order and check against the threshold
        #print holding_df.loc[d,list_symbols],prices_df.loc[d,list_symbols]
        leverage=sum(holding_df.loc[d,list_symbols].abs()*prices_df.loc[d,list_symbols])/(sum(holding_df.loc[d,list_symbols]*prices_df.loc[d,list_symbols])+holding_df.loc[d,'Cash'])
        # Check if the threshold has crossed 
        if leverage > leverage_threshold:
            # Then the order would not be processed
            # print d, leverage
            # update the holdings and trades_df based on BUY or SELL order
            holding_df.loc[d:,rows['Symbol']]-=trades_df.loc[d,rows['Symbol']] #This is same for both sell and buy
            
            if rows['Order']=='BUY':
                # First update the cash values back to the originals before the order
                trades_df.loc[d,'Cash']+=trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']]; 
                holding_df.loc[d:,'Cash']+=trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']];
                # Then update the trade count (Warning: doing this first will not change the trades and holding back to its inital values)
                trades_df.loc[d,rows['Symbol']]+=-rows['Shares']
                # print trades_df.loc[d,rows['Symbol']]
                rejected_order='BUY'
                
            elif rows['Order']=='SELL': # Negative value indicates you are selling
                # First update cash values
                trades_df.loc[d,'Cash']+=trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']]; # This will be +ve since we are selling the stock and it adds to the cash value
                holding_df.loc[d:,'Cash']+=trades_df.loc[d,rows['Symbol']]*prices_df.loc[d,rows['Symbol']];
                # Second update the trade count
                trades_df.loc[d,rows['Symbol']]+=rows['Shares']
                # print trades_df.loc[d,rows['Symbol']]
                rejected_order='SELL'
                
            print ('This {0} order on {1} for {2} stocks has been rejected due to leverage of {3}'.format(rejected_order,rows['Symbol'],rows['Shares'],leverage));
    
    values_df=prices_df*holding_df; # * in pandas gives cell-by-cell multiplication of dataframes
    # print '************'
    # print values_df
    stocks_df=values_df[list_symbols]
    
    #leverage_df=stocks_df.abs().sum(axis=1)/(stocks_df.sum(axis=1)+values_df['Cash']) # But you don't need this (since we calculate leverage per order basis)
    # Change it to dataframe with renaming 
    #leverage_df=pd.DataFrame(leverage_df.ix[:],index=stocks_df.index,columns=['Leverage'])
    
    portval=pd.DataFrame(values_df.sum(axis=1),index=values_df.index,columns=['Portval'])
    
    param_df=pd.DataFrame(index=['Portval','Benchmark ('+app_marketsim.vars['bench_sym']+')'],columns=['Sharpe ratio','Cumulative Return', 
                                                                                                       'Standard Deviation','Average Daily Return'])
    
    
    # Compute all the fund parameters:
    sharpe_ratio,cum_ret,std_daily_ret,avg_daily_ret=compute_indicators(portval['Portval']) # for this function to work, pass only the portval column
    param_df.ix['Portval']=[sharpe_ratio,cum_ret,std_daily_ret,avg_daily_ret]
           
    # Compute benchmark parameters
    
    sharpe_ratio_bench,cum_ret_bench,std_daily_ret_bench,avg_daily_ret_bench=compute_indicators(all_prices_df[app_marketsim.vars['bench_sym']])
    param_df.ix['Benchmark ('+app_marketsim.vars['bench_sym']+')']=[sharpe_ratio_bench,cum_ret_bench,std_daily_ret_bench,avg_daily_ret_bench]
    
    #print param_df
    
    return portval,all_prices_df,param_df


def compute_indicators(portvals,sf=252.0,rfr=0.0):
   
    # Compute daily returns
    daily_returns=util.compute_daily_returns(portvals)
    
    # compute cumulative daily return for portfolio
    cr=util.cummulative_return(portvals)
    
    # average daily return
    adr=daily_returns.mean()    
    
    # standard deviation of the daily return
    sddr= daily_returns.std()
    
    # Sharpe ratio
    sr=util.sharpe_ratio(adr, sddr, sf, rfr)
        
    return sr,cr,sddr,adr
