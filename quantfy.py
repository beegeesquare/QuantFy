from flask import Flask,render_template,redirect,request

from collections import defaultdict

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.embed import components
from bokeh.palettes import Spectral11
from bokeh.charts import Line,Area
from bokeh.layouts import gridplot

import pandas as pd
import datetime as dt
import util
import apicall_data
import optimization



from price_plot_dynamic import plot_stock_prices_1
from price_plot_dynamic_2 import plot_stock_prices_2
from martket_simulator import marketSim

# import bokeh
# print (bokeh.__version__)

app_quantfy=Flask(__name__)

app_quantfy.register_blueprint(plot_stock_prices_1)
app_quantfy.register_blueprint(plot_stock_prices_2)
app_quantfy.register_blueprint(marketSim)


@app_quantfy.route('/welcome_page_quantfy')
def welcome_page_quantfy():
    return "Welcome to Quantfy"

@app_quantfy.route('/') # redirect the page to index page if nothing is mentioned
def main():
    return redirect('/index')

@app_quantfy.route('/index',methods=['GET'])
def index():
     # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
    
    return render_template('index.html') # This comes when the website address is requested    
   
        



@app_quantfy.route('/price_plot',methods=['GET','POST'])
def plot_stock_prices():
    if request.method=='GET':
        return render_template('plot_prices.html')
    else:
        
        app_quantfy.vars={} # This is a dictionary
        # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
        
        app_quantfy.vars['sym'] = request.form['sym'].upper().strip(';').split(';') # 'sym' should be defined in html file as name
                
        if 'data' in request.form:
            app_quantfy.vars['data_src']=request.form['data']; # This get the data source
        else:
           
            return render_template('plot_prices.html',error_data='<font size="3" color="red" > Choose at least one data source </font>') 

        #print request.form
        
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_quantfy.vars['start_date']=dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 5 years ago of the current date
            app_quantfy.vars['start_date']=dt.datetime.today()-dt.timedelta(days=5*365) # This does not give the accurate 5 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_quantfy.vars['end_date']=dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_quantfy.vars['end_date']=dt.datetime.today()
        
        
        app_quantfy.vars['price_features']=defaultdict(); # This is an empty dictionary
        
        app_quantfy.vars['compute_features']=defaultdict(); # This is a dictionary for computing the requested features
               
        if 'closeprice' in request.form: app_quantfy.vars['price_features']['Close']=request.form['closeprice'] # Keys here should be in the same form that we get from database (like quandl)
                   
        if 'adjclose' in request.form:  app_quantfy.vars['price_features']['Adj. Close']=request.form['adjclose']
        
        if 'openprice' in request.form: app_quantfy.vars['price_features']['Open']=request.form['openprice']
        
        if 'adjopen' in request.form: app_quantfy.vars['price_features']['Adj. Open']=request.form['adjopen']
        
        if 'bench_sym' in request.form: 
            app_quantfy.vars['bench_sym']=request.form['bench_sym']
        else:
            app_quantfy.vars['bench_sym']='SPY'
        
        # These are the parameters that needs to be computed for each symbol. For this to be computed we need adjusted closing price
        
        if 'rollingmean' in request.form: app_quantfy.vars['compute_features']['RM']=request.form['rollingmean'] 
                   
        if 'rollingstd' in request.form:  app_quantfy.vars['compute_features']['RSTD']=request.form['rollingstd']
        
        if 'bollingerbands' in request.form: app_quantfy.vars['compute_features']['BOLLBNDS']=request.form['bollingerbands']
        
        if 'dailyreturns' in request.form: app_quantfy.vars['compute_features']['DR']=request.form['dailyreturns']
        
        if 'cummdailyreturns' in request.form: app_quantfy.vars['compute_features']['CDR']=request.form['cummdailyreturns']
        
        if 'avgdailyreturns' in request.form: app_quantfy.vars['compute_features']['ADR']=request.form['avgdailyreturns']
        
        if 'sharperatio' in request.form: app_quantfy.vars['compute_features']['SR']=request.form['sharperatio']
        
               
        
        if (app_quantfy.vars['sym'][0]=='') :  # sym is a list delimited by ;
            return render_template('plot_prices.html',error_sym='<font size="3" color="red" > Provide at least one ticker symbol </font>') 
               
        if len(app_quantfy.vars['price_features'])==0:
            return render_template('plot_prices.html',error_features='<font size="3" color="red"> At least one feature has to be selected </font>')
        
        
        symbols=app_quantfy.vars['sym'] # Here symbol will be a list
    
        # Add the benchmark symbol to the symbols
        symbols.insert(0,app_quantfy.vars['bench_sym']); # Insert the default symbol in the symbols list
        
        usr_price_features=list(app_quantfy.vars['price_features'].keys()) 
        
        #print symbols,usr_price_features, app_quantfy.vars['start_date'],app_quantfy.vars['end_date']
        
        if app_quantfy.vars['data_src']=='get_data_quandl':   
             # This is list of tuples, with first element in the tuple being symbol and second element being dict
             # Here get the data for all the price features, filter it in the plot_symbols function, so all the price features are present here
            full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, start_dt=app_quantfy.vars['start_date'],end_dt=app_quantfy.vars['end_date'])
                        ) for sym in symbols]
            
            # Pass user selected price features to the plot function
            script_ele,div_ele=plot_symbols(full_data,usr_price_features, 'Quandl')
            
            computed_df=compute_params(full_data)
            
            script_param,div_param=plot_params(full_data)
            
            return render_template('plot_prices.html',script_symbols=script_ele,plot_div_symbols=div_ele,computed_params=computed_df.to_html(),
                                   script_params=script_param,plot_div_features=div_param)
        
        elif app_quantfy.vars['data_src']=='get_data_yahoo': 
            # TODO: This needs a lot of change (pass not yet implemented)
            # data_dict=apicall_data.get_data_from_yahoo(symbol=symbols, features=features)
            return render_template('plot_prices.html',error_yahoo='<font size="3" color="red"> Data source not yet enabled </font>')
        
        # Here when the user clicks submit button, the bokeh plot should be displayed
        
@app_quantfy.route('/portfolio',methods=['GET','POST'])
def portfolio_page():
    if request.method=='GET':
        return render_template('portfolio.html')
    else:
        app_quantfy.vars={} # This is a dictionary
        # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
        
        app_quantfy.vars['sym'] = request.form['sym'].upper().strip(';').split(';') # 'sym' should be defined in html file as name
        
        if (app_quantfy.vars['sym'][0]=='') :  # sym is a list delimited by ;
            return render_template('portfolio.html',error_sym='<font size="3" color="red" > Provide at least one ticker symbol </font>') 
        
        
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_quantfy.vars['start_date']=dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('portfolio.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 1 years ago of the current date
            app_quantfy.vars['start_date']=dt.datetime.today()-dt.timedelta(days=365) # This does not give the accurate 5 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_quantfy.vars['end_date']=dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('portfolio.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_quantfy.vars['end_date']=dt.datetime.today()
        
        #print app_quantfy.vars
        if 'bench_sym' in request.form: 
            app_quantfy.vars['bench_sym']=request.form['bench_sym']
        else:
            app_quantfy.vars['bench_sym']='SPY'
        
        symbols=list(app_quantfy.vars['sym']); # Create a new list as we are doing insert operation next
        symbols.insert(0,app_quantfy.vars['bench_sym']); # Insert the default symbol in the symbols list
        
        # Here just get the data for the 'Adj. Close'
        full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=['Adj. Close'], start_dt=app_quantfy.vars['start_date'],end_dt=app_quantfy.vars['end_date'])
                        ) for sym in symbols]
        
        # Convert this to required format
        
        df_all_sym=util.get_data(full_data)
        
        app_quantfy.vars['guess_alloc']=request.form['guess_alloc'].strip(';').split(';')
        
        
        
        app_quantfy.vars['start_value']=float(request.form['start_value']); # It has a default value
        
        if len(app_quantfy.vars['guess_alloc']) !=0 and (app_quantfy.vars['guess_alloc'][0]!='') : # app_quantfy.vars['guess_alloc'] is a list because of the strip function
            print app_quantfy.vars['guess_alloc']
            print len(app_quantfy.vars['guess_alloc'])
            app_quantfy.vars['guess_alloc']=[float(i) for i in app_quantfy.vars['guess_alloc']]
            try:
                assert len(app_quantfy.vars['guess_alloc'])==len(app_quantfy.vars['sym'])
            except AssertionError:
                return render_template('portfolio.html',error_alloc='<font size="3" color="red" > Number of allocations should be same as symbols   </font>')
            # Sum should be equal to one
            print app_quantfy.vars['guess_alloc']
            
            try:
                assert abs(sum(app_quantfy.vars['guess_alloc'])-1.0)<=1e-5 # Sometimes the rounding does not work correctly
            except AssertionError:
                return render_template('portfolio.html',error_alloc='<font size="3" color="red" > Sum should be 1   </font>')

            
        else:
            # Generate random numbers
            allocs=np.random.random(len(app_quantfy.vars['sym']))
            allocs /=allocs.sum()
            app_quantfy.vars['guess_alloc']=allocs
            #print allocs
        
        cr,adr,sddr,sr,ev,normalized_plot_df=optimization.access_portfolio(df_all_sym, app_quantfy.vars['bench_sym'], 
                                                                           app_quantfy.vars['guess_alloc'],
                                                                           sv=app_quantfy.vars['start_value'])
        
        #print cr,adr,sddr,sr,ev
        
        param_not_opt=pd.DataFrame([cr,adr,sddr,sr,ev],index=['CR','ADR','STDDR','SR','EV'], columns=['Unoptimized'])
        
        #print normalized_plot_df.head()
        
        TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select'
        not_opt_p = figure(width=600, height=300, x_axis_type="datetime",tools=TOOLS)
        
        colors=['blue','red','green','#cc3300']
        
        for (i,ftr) in enumerate(normalized_plot_df):
            not_opt_p.line(normalized_plot_df.index,normalized_plot_df[ftr],legend=ftr,color=colors[i])
        
        #not_opt_p.line(normalized_plot_df)
        
        not_opt_p.title.text = "Un optimized portfolio value"
        not_opt_p.legend.location = "top_left"
        not_opt_p.grid.grid_line_alpha=0
        not_opt_p.xaxis.axis_label = 'Date'
        not_opt_p.yaxis.axis_label = 'Relative portfolio value'
        not_opt_p.ygrid.band_fill_color="olive"
        not_opt_p.ygrid.band_fill_alpha = 0.1
        
        script_not_opt, div_not_opt=components(not_opt_p)
        
        # print script_not_opt,div_not_opt
        # Now run optimized
        
        cr,adr,sddr,sr,ev,normalized_plot_df,optimal_alloc=optimization.optimize_portfolio(df_all_sym,app_quantfy.vars['bench_sym'],
                                                                             app_quantfy.vars['start_value'])
        
        
        # print cr,adr,sddr,sr,ev,optimal_alloc
        
        # print normalized_plot_df.head()
        
        opt_p = figure(width=600, height=300, x_axis_type="datetime",tools=TOOLS)
              
        for (i,ftr) in enumerate(normalized_plot_df):
            opt_p.line(normalized_plot_df.index,normalized_plot_df[ftr],legend=ftr,color=colors[i])
        
        #not_opt_p.line(normalized_plot_df)
        
        opt_p.title.text = "Optimized portfolio value"
        opt_p.legend.location = "top_left"
        opt_p.grid.grid_line_alpha=0
        opt_p.xaxis.axis_label = 'Date'
        opt_p.yaxis.axis_label = 'Relative portfolio value'
        opt_p.ygrid.band_fill_color="olive"
        opt_p.ygrid.band_fill_alpha = 0.1
        
        script_opt, div_opt=components(opt_p)
        
        param_opt=pd.DataFrame([cr,adr,sddr,sr,ev],index=['CR','ADR','STDDR','SR','EV'], columns=['Optimized'])
        
              
        alloc_df=pd.DataFrame([app_quantfy.vars['guess_alloc'],list(optimal_alloc)],index=['Random/Guess allocations','Optimized allocations'],columns=app_quantfy.vars['sym'])
        
        #str_opt_alloc='Optimal allocations: '+', '.join([str(i) for i in optimal_alloc])
        
        
        
        return render_template('portfolio.html',not_opt=param_not_opt.to_html(),opt=param_opt.to_html(), opt_alloc=alloc_df.to_html(),
                               script_not_opt=script_not_opt,plot_not_opt=div_not_opt, 
                               script_opt=script_opt,plot_opt=div_opt
                               )
    
     
    

def plot_symbols(list_data_tuples,usr_price_features,data_src):
    """
    Input: List of tuples where (x[0] is a symbol, x[1] is a dict) and data-source
    This function returns the div element of all symbols, for given features.
    So this div element contains plots==len(symbols), 
    """
    #script_el_data=''
    #div_el_data='<h4> Time-series plot for all symbols with chosen features </h4><table style="width 50%"> <tr>'
    list_plots=[]
    for tpl in list_data_tuples:
        
        if 'error' not in tpl[1]:
            full_df,features=tpl[1]['data'],tpl[1]['features']
            
            # Before plotting remove the features that are not necessary (i.e, use the features that user requested for)
            # Make sure all the user requested features are there in the dataframe.
            #print usr_price_features, features,tpl[0]
            
            plot_price_features=list(set(usr_price_features).intersection(set(features)))
            
            df=full_df[plot_price_features]
            
            
            TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select'
            p = figure(width=500, height=300, x_axis_type="datetime",tools=TOOLS)
            colors=['blue','red','green','#cc3300']
            
            assert(len(colors)>=len(features))
            
            for (i,ftr) in enumerate(plot_price_features):
                p.line(df.index,df[ftr],legend=ftr,color=colors[i])
            
            p.title.text = "Data for %s from %s data source"%(tpl[0],data_src)
            p.legend.location = "top_left"
            p.grid.grid_line_alpha=0
            p.xaxis.axis_label = 'Date'
            p.yaxis.axis_label = 'Price'
            p.ygrid.band_fill_color="olive"
            p.ygrid.band_fill_alpha = 0.1
            
            list_plots.append(p)
            # Here I need to get the javascript for the plot and put it in the plot_features.html  (boilerplate) template
            #plt_script,plt_div=components(p) # script and div needs to be inserted in the plot_features.html
            
            #script_el_data+=plt_script; # This will be java script data element this need not be in a table form
            
            #div_el_data+='<td>'+plt_div+'</td>'
            #print (plt_script)
            #print (plt_div)
        else:
            print tpl[0]
            
            #div_el_data+='<td>'+'Ticker symbol %s not found in the database'%(tpl[0])+'</td>'

    #div_el_data+='</tr></table>'
    
    #print script_el_data, div_el_data       
    if len(list_plots)!=0:
        script_el_data, div_el_data=components(gridplot(list_plots,ncols=2, plot_width=600, plot_height=400,tools=TOOLS))
    else:
        script_el_data=''
        div_el_data=''
    
    
    return script_el_data, div_el_data

def compute_params(list_data_tuples):
    
    df=util.get_data(list_data_tuples) # Here columns will be the list of symbols as requested by user
    
    # Now compute the parameters and for each symbol in this above dataframe 
    
    computed_df=pd.DataFrame(index=['CDR','ADR','STDDR','SR'],columns=df.columns)
    
    for sym in df.columns:
        dr=util.compute_daily_returns(df[sym])
        
        cdr=util.cummulative_return(df[sym])
        adr=dr.mean()
        sddr=np.std(dr)
        sr=util.sharpe_ratio(adr, sddr)
        
        computed_df.ix['CDR'][sym]=cdr
        computed_df.ix['ADR'][sym]=adr
        computed_df.ix['STDDR'][sym]=sddr
        computed_df.ix['SR'][sym]=sr
    
    return computed_df

def plot_params(list_data_tuples):
    
    #script_el_param=''
    #div_el_param='<h4> Time-series plot for computed parameters <table style="width 50%">  <tr>'
    
    list_plots=[]
       
      
    # Here we need to compute the following:
    # (1) Daily returns (2)  Rolling standard deviation (3) Bollinger bands (4) Rolling std
    
    # Generate plot for each symbols
    
    df=util.get_data(list_data_tuples)
    # Normalize the data
    df=df/df.ix[0,:]
       
    daily_returns = util.compute_daily_returns(df)
    rolling_mean=util.get_rolling_mean(df, window=20)
    rolling_std=util.get_rolling_std(df, window=20)
    u_bollinger_bnd,l_bollinger_bnd=util.get_bollinger_bands(rolling_mean, rolling_std)
    
    param_dict={'Rolling mean':rolling_mean,'Rolling SD': rolling_std,'Bollinger Bands':(u_bollinger_bnd,l_bollinger_bnd),'Daily returns':daily_returns}
    
    #print Paired
    colors=Spectral11[0:len(daily_returns.columns)] # Paired is dict with keys as the number of colors needed. Largest Key is 12
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select'

    for param in param_dict.keys():
        p = figure(width=500, height=300, x_axis_type="datetime",tools=TOOLS)
    
        if param =="Bollinger Bands":
            upper_band=param_dict[param][0]
            lower_band=param_dict[param][1]
            for (i,sym) in enumerate(upper_band):
                p.line(upper_band.index,upper_band[sym],legend=sym,color=colors[i],line_width=2)
            
            for (i,sym) in enumerate(lower_band):
                p.line(lower_band.index,lower_band[sym],legend=sym,color=colors[i],line_width=2,line_dash='dashed')   
        else:
            for (i,sym) in enumerate(param_dict[param]):
                p.line(param_dict[param].index,param_dict[param][sym],legend=sym,color=colors[i],line_width=2)
                
        p.title.text = param
        p.legend.location = "top_left"
        p.grid.grid_line_alpha=0
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = param
        p.ygrid.band_fill_color="olive"
        p.ygrid.band_fill_alpha = 0.1
    
        # Here I need to get the javascript for the plot and put it in the plot_features.html  (boilerplate) template
        #plt_script,plt_div=components(p) # script and div needs to be inserted in the plot_features.html
        
        #script_el_param+=plt_script; # This will be java script data element this need not be in a table form
        
        #div_el_param+='<td>'+plt_div+'</td>'
        
        list_plots.append(p)
    
    if len(list_plots)!=0:
        script_el_param, div_el_param=components(gridplot(list_plots,ncols=2, plot_width=600, plot_height=400,tools=TOOLS))
    else:
        script_el_param=''
        div_el_param=''
    
    return script_el_param, div_el_param





@app_quantfy.route('/error',methods=['GET'])
def err():
    return render_template('error.html')

# TODO: Return to the error page if the symbol is unidentified
@app_quantfy.route('/symbol_error',methods=['GET'])
def error_symbol():
    return render_template('symbol_error.html')



@app_quantfy.route('/invest_trade',methods=['GET','POST'])
def trading_page():
    return render_template('invest_trade.html')


    

if __name__=='__main__':
    import os
    
    port=int(os.environ.get("PORT",5000)) # This is for heroku app to run I guess
    app_quantfy.run(host='0.0.0.0',debug=True,port=port)
    
    
    
    #http_server = HTTPServer(WSGIContainer(app_quantfy))
    #http_server.listen(5000)

    #io_loop.add_callback(view, "http://127.0.0.1:5000/")
    
    
    # df=get_data_from_quandl(symbol='SPY',features=['Close','Adj. Close','Open','Adj. Open'])
    #print (df.head())
    #print (df.tail())
    
    