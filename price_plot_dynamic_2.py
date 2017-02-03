from flask import Flask,render_template,redirect,request, Blueprint

from collections import defaultdict

import numpy as np

from bokeh.plotting import figure, show,output_file,ColumnDataSource
from bokeh.palettes import viridis
from bokeh.embed import components,autoload_static,autoload_server
from bokeh.layouts import gridplot,row, column, widgetbox
from bokeh.models.widgets import PreText, Select
from bokeh.models import HoverTool
from bokeh.io import curdoc,output_file

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler

from bokeh.charts import Area



import pandas as pd
import datetime as dt
import util
import apicall_data
import new



app_plots_2=Flask(__name__)






plot_stock_prices_2 = Blueprint('plot_prices_beta_2.html', __name__, template_folder='templates')

@plot_stock_prices_2.route('/price_plot_interactive_2',methods=['GET','POST'])
def plot_stock_prices_interactive_2():
    if request.method=='GET':
        return render_template('plot_prices_beta_2.html')
    else:
        
        app_plots_2.vars={} # This is a dictionary
        # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
        if 'data' in request.form:
            app_plots_2.vars['data_src']=request.form['data']; # This get the data source
        else:
           
            return render_template('plot_prices_beta_2.html',error_data='<font size="3" color="red" > Choose at least one data source </font>') 
        
        
        app_plots_2.vars['sym'] = request.form['sym'].upper().strip(';').split(';') # 'sym' should be defined in html file as name
        
        #print request.form
        if (app_plots_2.vars['sym'][0]=='') :  # sym is a list delimited by ;
            return render_template('plot_prices_beta_2.html',error_sym='<font size="3" color="red" > Provide at least one ticker symbol </font>') 
        
               
        
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_plots_2.vars['start_date']=dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices_beta_2.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 1 year ago of the current date
            app_plots_2.vars['start_date']=dt.datetime.today()-dt.timedelta(days=365) # This does not give the accurate 5 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_plots_2.vars['end_date']=dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices_beta_2.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_plots_2.vars['end_date']=dt.datetime.today()
        
        if 'bench_sym' in request.form: 
            app_plots_2.vars['bench_sym']=request.form['bench_sym']
        else:
            app_plots_2.vars['bench_sym']='SPY'
              
        symbols=app_plots_2.vars['sym'] # Here symbol will be a list
        
        # print request.form
          
        # Add the benchmark symbol to the symbols
        symbols.insert(0,app_plots_2.vars['bench_sym']); # Insert the default symbol in the symbols list       
        
        if app_plots_2.vars['data_src']=='get_data_quandl':   
            # Here get all the required features
            
            features=['Open','Close','High','Low','Volume','Adj. Close']
            
            full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=features,start_dt=app_plots_2.vars['start_date'],end_dt=app_plots_2.vars['end_date'])
                        ) for sym in symbols]
            
            
            script_ele,div_ele=plot_symbols_interactive_2(full_data,'Quandl')
            
            #computed_df=compute_params(full_data)
            
            #script_param,div_param=plot_params(full_data)
            
            return render_template('plot_prices_beta_2.html',script_symbols=script_ele,plot_div_symbols=div_ele
                                  )
        
        elif app_plots_2.vars['data_src']=='get_data_yahoo': 
            # TODO: This needs a lot of change (pass not yet implemented)
            # data_dict=apicall_data.get_data_from_yahoo(symbol=symbols, features=features)
            return render_template('plot_prices_beta_2.html',error_yahoo='<font size="3" color="red"> Data source not yet enabled </font>')
        
        # Here when the user clicks submit button, the bokeh plot should be displayed


def plot_symbols_interactive_2(list_data_tuples,data_src):
    """
    Input: List of tuples where (x[0] is a symbol, x[1] is a dict) and data-source
    This function returns the div element of all symbols, for given features.
    So this div element contains plots==len(symbols), 
    """
    script_el_data=''
    div_el_data=''
    
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
    
    
    # set up widgets
    price_features=['Adj. Close','Open','High','Low','Close']
    
    
    feature_ticker=Select(title='Price Features',value=price_features[0],options=price_features);
    symbol_ticker= Select(title='Ticker symbols',value=list_symbols[1],options=list_symbols[1:]); # As SPY is the first element of the list (choose the user selected feature)
    
    
    # set up plots
    source_static=ColumnDataSource(data=dict(Date=[], t1=[]))
    # benchmark line (this remains the same)
    benchSym=list_symbols[0];
    source_benchmark=ColumnDataSource(data=dict(Date=adj_prices_df.index,t2=adj_prices_df[benchSym]))
    
    
    p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Stock price'
    
    p.line('Date','t1',source=source_static,line_width=2)
    p.line('Date','t2',source=source_benchmark,line_width=2,color='red')
      
    # Set up callbacks  
    
    def callback(attrname,old,new):
        # Take the new value and update the graph with the new value
        update(new)
        
    
    def update(new=list_symbols[1]):
        # Get the data for the new ticker symbol
        print new
        source_static.data=dict(Date=adj_prices_df.index,t1=adj_prices_df[new])
        
        
    feature_ticker.on_change('value',callback)
    
    # set up layout
    widgets=column(feature_ticker, symbol_ticker)

    layout = row(p,widgets)
    
    # initialize
    update()
        
    curdoc().add_root(layout)   
    
    #print curdoc()
    
    # Get the default symbol from the list_data_tuples
    
    '''
    ListPltData=list_data_tuples[0:2]; # Take the first and second element in the list, first is a benchmark symbol
    
    for (i,tpl) in enumerate(ListPltData):

        dfAllFeatures=tpl[1]['data'] # df for all features for a particular symbol
        
        dfAllFeatures['Volume']=dfAllFeatures['Volume']/1.e6
         
        hover_source=ColumnDataSource(data=dfAllFeatures)
        
        p.line('Date','Adj. Close',line_width=2,source=hover_source,legend=tpl[0],color=colors[i]) 
        
    '''
    
     
    
    script_el_data, div_el_data=components(layout)
    
    
    
    return script_el_data, div_el_data

