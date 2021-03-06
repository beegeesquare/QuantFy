from flask import Flask,render_template,redirect,request

from collections import defaultdict

import numpy as np
from bokeh.plotting import figure, show,ColumnDataSource
from bokeh.io import output_file
from bokeh.embed import components
from bokeh.palettes import Spectral11,viridis
from bokeh.charts import Line,Area
from bokeh.layouts import gridplot,WidgetBox
from bokeh.models import HoverTool, Span, Label
from bokeh.models.widgets import Panel, Tabs, DataTable,TableColumn


import pandas as pd
import datetime as dt
import util
import apicall_data
import optimization



from price_plot_dynamic import plot_stock_prices_1
from price_plot_dynamic_2 import plot_stock_prices_2
from martket_simulator import marketSim
from ml_models import mlModels
from commodity import commodity

# import bokeh
# print (bokeh.__version__)

app_quantfy=Flask(__name__)

app_quantfy.register_blueprint(plot_stock_prices_1)
app_quantfy.register_blueprint(plot_stock_prices_2)
app_quantfy.register_blueprint(marketSim)
app_quantfy.register_blueprint(mlModels)
app_quantfy.register_blueprint(commodity)

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
               
        
        
        if 'bench_sym' in request.form: 
            app_quantfy.vars['bench_sym']=request.form['bench_sym']
        else:
            app_quantfy.vars['bench_sym']='SPY'
        
        
        if (app_quantfy.vars['sym'][0]=='') :  # sym is a list delimited by ;
            return render_template('plot_prices.html',error_sym='<font size="3" color="red" > Provide at least one ticker symbol </font>') 
               
        
        symbols=app_quantfy.vars['sym'] # Here symbol will be a list
    
        # Add the benchmark symbol to the symbols
        symbols.insert(0,app_quantfy.vars['bench_sym']); # Insert the default symbol in the symbols list
        
        usr_price_features=list(app_quantfy.vars['price_features'].keys()) 
        
        #print symbols,usr_price_features, app_quantfy.vars['start_date'],app_quantfy.vars['end_date']
        
        if app_quantfy.vars['data_src']=='get_data_quandl':   
            # This is list of tuples, with first element in the tuple being symbol and second element being dict
            # Here get the data for all the price features, filter it in the plot_symbols function, so all the price features are present here
            features=['Open','Close','High','Low','Volume','Adj. Close']
            
            full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=features,start_dt=app_quantfy.vars['start_date'],end_dt=app_quantfy.vars['end_date'])
                        ) for sym in symbols]
            
            # Pass user selected price features to the plot function
            #script_ele,div_ele=plot_symbols(full_data,usr_price_features, 'Quandl')
            
            script_ele,div_ele=plot_symbols_interactive(full_data, 'Quandl')
            
            computed_df,describe_df=compute_params(full_data)
            
            # print describe_df
            computed_df=computed_df.round(5)
            
            script_computed_params,div_computed_params=convert_pd_bokeh_html(computed_df.round(4))
            script_describe,div_describe=convert_pd_bokeh_html(describe_df.round(4))
            
            script_param,div_param=plot_params(full_data)
            
            return render_template('plot_prices.html',script_symbols=script_ele,plot_div_symbols=div_ele,
                                   script_computed_params=script_computed_params,div_computed_params=div_computed_params,
                                   script_params=script_param,plot_div_features=div_param,
                                   script_describe=script_describe,div_describe=div_describe)
        
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
            # print app_quantfy.vars['guess_alloc']
            # print len(app_quantfy.vars['guess_alloc'])
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
        
        param_not_opt=pd.DataFrame([cr,adr,sddr,sr,ev],index=['Cumulative Return','Average Daily Return','Stand. Deviation Daily return',
                                                          'Sharpe Ratio','End value'], columns=['Unoptimized'])
        
        script_not_opt_table,div_not_opt_table=convert_pd_bokeh_html(param_not_opt)
        
        # print normalized_plot_df.head()
        hover=HoverTool(
            tooltips=[
                ("Portfolio",'$y')
                
                
            ]
        )
        TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
        not_opt_p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
        
        colors=['blue','red','green','#cc3300']
        
        for (i,ftr) in enumerate(normalized_plot_df):
            not_opt_p.line(normalized_plot_df.index,normalized_plot_df[ftr],legend=ftr,color=colors[i],line_width=2)
        
        #not_opt_p.line(normalized_plot_df)
        
        not_opt_p.title.text = "Un-optimized portfolio"
        not_opt_p.legend.location = "top_left"
        not_opt_p.xaxis.axis_label = 'Date'
        not_opt_p.yaxis.axis_label = 'Relative portfolio value'
        
        tab_not_opt=Panel(child=not_opt_p,title='Un-optimized portfolio')
        
        # script_not_opt, div_not_opt=components(not_opt_p)
        
        # print script_not_opt,div_not_opt
        # Now run optimized
        
        cr,adr,sddr,sr,ev,normalized_plot_df,optimal_alloc=optimization.optimize_portfolio(df_all_sym,app_quantfy.vars['bench_sym'],
                                                                             app_quantfy.vars['start_value'])
        
        
        # print cr,adr,sddr,sr,ev,optimal_alloc
        
        # print normalized_plot_df.head()
        hover=HoverTool(
            tooltips=[
                ("Portfolio",'$y')
                
                
            ]
        )
        
        opt_p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
              
        for (i,ftr) in enumerate(normalized_plot_df):
            opt_p.line(normalized_plot_df.index,normalized_plot_df[ftr],legend=ftr,color=colors[i],line_width=2)
        
        
        # print normalized_plot_df
        opt_p.title.text = "Optimized portfolio value"
        opt_p.legend.location = "top_left"
        opt_p.xaxis.axis_label = 'Date'
        opt_p.yaxis.axis_label = 'Relative portfolio value'
        
        tab_opt=Panel(child=opt_p,title='Optimized portfolio')
        
        tabs=Tabs(tabs=[tab_not_opt,tab_opt])
        
        script_opt, div_opt=components(tabs)
        
        
        param_opt=pd.DataFrame([cr,adr,sddr,sr,ev],index=['Cummulative Return','Additive Daily Return','Stand. Deviation Daily return',
                                                          'Sharpe Ratio','End value'], columns=['Optimized'])
        
        all_params=param_not_opt.join(param_opt)
        
        script_opt_table,div_opt_table=convert_pd_bokeh_html(all_params)
        
        
              
        alloc_df=pd.DataFrame([app_quantfy.vars['guess_alloc'],list(optimal_alloc)],index=['Random/Guess allocations','Optimized allocations'],columns=app_quantfy.vars['sym'])
        
        #str_opt_alloc='Optimal allocations: '+', '.join([str(i) for i in optimal_alloc])
        script_alloc_df,div_alloc_df=convert_pd_bokeh_html(alloc_df)
        
        # script_not_opt_table=script_not_opt_table,div_not_opt_table=div_not_opt_table,
        return render_template('portfolio.html',script_opt_table=script_opt_table, div_opt_table=div_opt_table,
                               script_alloc_df=script_alloc_df,div_alloc_df=div_alloc_df,
                                script_opt=script_opt,plot_opt=div_opt
                               )
    
     

def plot_symbols_interactive(list_data_tuples,data_src):
    """
    Input: List of tuples where (x[0] is a symbol, x[1] is a dict) and data-source
    This function returns the div element of all symbols, for given features.
    So this div element contains plots==len(symbols), 
    """
    #script_el_data=''
    #div_el_data='<h4> Time-series plot for all symbols with chosen features </h4><table style="width 50%"> <tr>'
    list_plots=[]
    usr_price_features=['Open','High','Low','Close','Volume','Adj. Close']
    # Just draw one plot for all the ticker symbols requested for
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
    
    hover=HoverTool(
            tooltips=[
                #("Date",'$x.strftime("%Y-%m-%d")'),
                ("Adj. Close",'$y'),
                ("Open", "@Open"),
                ("Close", "@Close"),
                ("Low", "@Low"),
                ("High","@High"),
                ("Volume (M)","@Volume")
            ]
        )
    
    script_el_data=''
    div_el_data=''
    # Create a data frame with adjusted close price as the main
    adj_prices_df=util.get_data(list_data_tuples)
    
    list_symbols=list(adj_prices_df.columns)
    
    colors=viridis(len(list_symbols))
    
    p = figure(width=1200, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
  
   
    #print colors
    
    # Create an area-plot for adjusted close price for each symbol
    
    for (i,tpl) in enumerate(list_data_tuples):
        
        if 'error' not in tpl[1]:
            dfAllFeatures=tpl[1]['data'] # df for all features for a particular symbol
            #print dfAllFeatures.columns
            dfAllFeatures['Volume']=dfAllFeatures['Volume']/1.e6
            # 
            source=ColumnDataSource(data=dfAllFeatures)
            #source=ColumnDataSource({'x':dfAllFeatures.index,'y':dfAllFeatures['Adj. Close'],})
            p.line('Date','Adj. Close',line_width=2,source=source,legend=tpl[0],color=colors[i]) 
        
    
        else:
            print "Symbol %s not found"%(tpl[0])
    
    p.title.text = "Data for requested %s ticker symbols from %s data source"%(", ".join(list_symbols),data_src)
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price (Adjusted Close)'
    
    script_el_data, div_el_data=components(p)
        
    return script_el_data, div_el_data

  

def plot_symbols(list_data_tuples,usr_price_features,data_src):
    """
    Input: List of tuples where (x[0] is a symbol, x[1] is a dict) and data-source
    This function returns the div element of all symbols, for given features.
    So this div element contains plots==len(symbols), 
    """
    #script_el_data=''
    #div_el_data='<h4> Time-series plot for all symbols with chosen features </h4><table style="width 50%"> <tr>'
    list_plots=[]
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
    colors=['blue','red','green','#cc3300']
    for tpl in list_data_tuples:
        
        if 'error' not in tpl[1]:
            full_df,features=tpl[1]['data'],tpl[1]['features']
            
            # Before plotting remove the features that are not necessary (i.e, use the features that user requested for)
            # Make sure all the user requested features are there in the dataframe.
            #print usr_price_features, features,tpl[0]
            
            plot_price_features=list(set(usr_price_features).intersection(set(features)))
            
            df=full_df[plot_price_features]
            hover=HoverTool(
            tooltips=[
                ("Price",'$y')
                ]
            )
            
            
            p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
            
            
            assert(len(colors)>=len(features))
            
            for (i,ftr) in enumerate(plot_price_features):
                p.line(df.index,df[ftr],legend=ftr,color=colors[i])
            
            p.title.text = "Data for %s from %s data source"%(tpl[0],data_src)
            p.legend.location = "top_left"
            p.xaxis.axis_label = 'Date'
            p.yaxis.axis_label = 'Price'
            
            tab=Panel(child=p,title=tpl[0])
            
            list_plots.append(tab)
        
        else:
            print tpl[0]
            
            #div_el_data+='<td>'+'Ticker symbol %s not found in the database'%(tpl[0])+'</td>'

    #div_el_data+='</tr></table>'
    
    #print script_el_data, div_el_data       
    if len(list_plots)!=0:
       script_el_data, div_el_data=components(Tabs(tabs=list_plots))
    else:
        script_el_data=''
        div_el_data=''
    
    
    return script_el_data, div_el_data

def compute_params(list_data_tuples):
    
    df=util.get_data(list_data_tuples) # Here columns will be the list of symbols as requested by user
    
    # Now compute the parameters and for each symbol in this above dataframe 
    
    # Also get the data description using the describe feature
    
    describe_features=df.describe()
    
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
    
    return computed_df,describe_features

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
    # drop the rows where the values are 0, for instance for window 20, the first 20 are zeros
    rolling_mean=rolling_mean[(rolling_mean.sum(axis=1)!=0)]   
    rolling_std=util.get_rolling_std(df, window=20)
    rolling_std=rolling_std[(rolling_std.sum(axis=1)!=0)]
    
    u_bollinger_bnd,l_bollinger_bnd=util.get_bollinger_bands(rolling_mean, rolling_std)
    
    param_dict={'Rolling mean':rolling_mean,'Rolling SD': rolling_std,'Bollinger Bands':(u_bollinger_bnd,l_bollinger_bnd),'Daily returns':daily_returns}
    
    colors=viridis(len(daily_returns.columns));
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
   

    for param in param_dict.keys():
        hover=HoverTool(
            tooltips=[
                ("Metric",'$y')
                ]
            )
        p = figure(width=1200, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
    
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
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = param
       
        tab=Panel(child=p,title=param)
        
        list_plots.append(tab)
    
    if len(list_plots)!=0:
        script_el_param, div_el_param=components(Tabs(tabs=list_plots))
    else:
        script_el_param=''
        div_el_param=''
    
    return script_el_param, div_el_param

def convert_pd_bokeh_html(df):
    
    # Put the metrics table in the html using bokeh
    df_data=dict(df[[i for i in df.columns]].round(4) )
   
    df_data['Metric']=df.index # This will add the index (Note: Instead of Metric, if I use index, then the width of output index column cannot be adjustested )
    source=ColumnDataSource(df_data)
    columns=[TableColumn(field=i,title=i) for i in df.columns]
    # Insert the index column to the list of columns
    columns.insert(0, TableColumn(field="Metric",title="Metric"))
    df_table=DataTable(source=source,columns=columns, height=200, width=450)
    table_script,table_div= components(WidgetBox(df_table))
    
    return table_script,table_div



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
    
    