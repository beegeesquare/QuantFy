from flask import Flask,render_template,redirect,request

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.embed import components
import pandas as pd
import datetime as dt
# import bokeh
# print (bokeh.__version__)

app_quantfy=Flask(__name__)





@app_quantfy.route('/welcome_page_quantfy')
def welcome_page_quantfy():
    return "Welcome to Quantfy"

@app_quantfy.route('/') # redirect the page to index page if nothing is mentioned
def main():
    return redirect('/index')

@app_quantfy.route('/index',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html') # This comes when the website address is requested
    else:
        
        # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
        app_quantfy.vars={} # This is a dictionary
        app_quantfy.vars['sym'] = request.form['sym'].upper() # 'sym' should be defined in html file as name
               
        if 'closeprice' in request.form: 
            app_quantfy.vars['Close']=request.form['closeprice'] # Keys here should be in the same form that we get from database (like quandl)
           
        
        if 'adjclose' in request.form:  app_quantfy.vars['Adj. Close']=request.form['adjclose']
        
        if 'openprice' in request.form: app_quantfy.vars['Open']=request.form['openprice']
        
        if 'adjopen' in request.form: app_quantfy.vars['Adj. Open']=request.form['adjopen']
        
        
        
        if len(app_quantfy.vars.keys())<2 or (app_quantfy.vars['sym']=='') : return redirect('/error') # Symbol cannot be empty
        
        # Here when the user clicks submit button, the bokeh plot should be displayed
        return redirect('/plot')

@app_quantfy.route('/plot',methods=['GET'])
def plot():
    #TODO: This has to be the bokeh plot
    # Using the values selected from the 
    # print (app_quantfy.vars)
    symbol=app_quantfy.vars['sym']
    features=list(app_quantfy.vars.keys()) # First being the symbol
    
    features.remove('sym')
    
    data_dict=get_data_from_quandl(symbol=symbol,features=features)
    
    if 'error' not in data_dict:
        df,features,data_src=data_dict['data'],data_dict['features'],data_dict['src']
    
        TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select'
        p = figure(width=800, height=500, x_axis_type="datetime",tools=TOOLS)
        colors=['blue','red','green','#cc3300']
        
        assert(len(colors)>=len(features))
        
        for (i,ftr) in enumerate(features):
            p.line(df.index,df[ftr],legend=ftr,color=colors[i])
        
        p.title.text = "Data for %s from Quandle %s set"%(symbol,data_src)
        p.legend.location = "top_left"
        p.grid.grid_line_alpha=0
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Price'
        p.ygrid.band_fill_color="olive"
        p.ygrid.band_fill_alpha = 0.1
        
        # Here I need to get the javascript for the plot and put it in the plot_features.html  (boilerplate) template
        plt_script,plt_div=components(p) # script and div needs to be inserted in the plot_features.html
        
        # print (plt_script)
        # print (plt_div)
        
        
        # sym, plot_script,  plot_div should be defined in the html file as {{}}
        return render_template('plot_features.html',sym=symbol, plot_script=plt_script,plot_div=plt_div) 
    
    else:
        return redirect('/symbol_error')

@app_quantfy.route('/error',methods=['GET'])
def err():
    return render_template('error.html')

# TODO: Return to the error page if the symbol is unidentified
@app_quantfy.route('/symbol_error',methods=['GET'])
def error_symbol():
    return render_template('symbol_error.html')

def get_data_from_quandl(symbol,features,start_dt=dt.datetime(2000,1,1),end_dt=dt.datetime.today()):
    """
    Gets the required data for the given symbol from quandl and store it as the csv file
    Store the file name as : {SYMBOL.csv}. This has some problems
    """
    
    import quandl
    quandl.ApiConfig.api_key = "MKXxiRmCQyr6ysZ5Qd2x"
    # For SPY use Yahoo instead of WIKI, but it will only give (Open, High,close, adjusted close, volume)
    if symbol=='SPY':
        
        dataframe=quandl.get('YAHOO/INDEX_SPY',start_date=start_dt.strftime('%Y-%m-%d'),
                             end_date=end_dt.strftime('%Y-%m-%d')) # This will be a pandas dataframe
        # Rename Adjusted Close to Adj_Close...I think
        dataframe=dataframe.rename(columns={'Adjusted Close':'Adj. Close'})
        data_source='YAHOO'
        # Return the dataframe only with the entries requested and remove the ones that are not present in YAHOO database
        df=pd.DataFrame(index=dataframe.index) # Create an empty dataframe with indices being dates
        removed_features=[]; # If the user selects the features that are not part of YAHOO data set
        for ftrs in features:
            try:
                df=df.join(dataframe[ftrs])
            except KeyError:
                removed_features.append(ftrs) # Returns the features to the plot that are not found in dataset
                
        for rm in removed_features: features.remove(rm)
        
        if len(features)==0:
            data_dict={'error':'Symbol not found'}
            return data_dict
        else:
            data_dict={'data':df,'features':features,'src':data_source}
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
       
    
    




if __name__=='__main__':
    import os
    port=int(os.environ.get("PORT",5000)) # This is for heroku app to run I guess
    app_quantfy.run(host='0.0.0.0',debug=True,port=port)
    # df=get_data_from_quandl(symbol='SPY',features=['Close','Adj. Close','Open','Adj. Open'])
    #print (df.head())
    #print (df.tail())
    
    