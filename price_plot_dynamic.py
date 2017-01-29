from flask import Flask,render_template,redirect,request, Blueprint

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



app_plots=Flask(__name__)

plot_stock_prices_2 = Blueprint('plot_prices_beta_1.html', __name__, template_folder='templates')
@plot_stock_prices_2.route('/price_plot_interactive',methods=['GET','POST'])
def plot_stock_prices_interactive():
    if request.method=='GET':
        return render_template('plot_prices_beta_1.html')
    else:
        
        app_plots.vars={} # This is a dictionary
        # Define the variables. This is a local variable, but in Flask it will be passed to the plot route I guess
        
        app_plots.vars['sym'] = request.form['sym'].upper().strip(';').split(';') # 'sym' should be defined in html file as name
                
        
        #print request.form
        
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_plots.vars['start_date']=dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices_beta_1.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 1 year ago of the current date
            app_plots.vars['start_date']=dt.datetime.today()-dt.timedelta(days=365) # This does not give the accurate 5 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_plots.vars['end_date']=dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('plot_prices_beta_1.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_plots.vars['end_date']=dt.datetime.today()
        
        if 'bench_sym' in request.form: 
            app_plots.vars['bench_sym']=request.form['bench_sym']
        else:
            app_plots.vars['bench_sym']='SPY'
              
        
        symbols=app_plots.vars['sym'] # Here symbol will be a list
        
        print request.form
        if 'data' in request.form:
            app_plots.vars['data_src']=request.form['data']; # This get the data source
        else:
           
            return render_template('plot_prices_beta_1.html',error_data='<font size="3" color="red" > Choose at least one data source </font>') 

    
        # Add the benchmark symbol to the symbols
        symbols.insert(0,app_plots.vars['bench_sym']); # Insert the default symbol in the symbols list
        
                
        #print symbols,usr_price_features, app_plots.vars['start_date'],app_plots.vars['end_date']
        
        if app_plots.vars['data_src']=='get_data_quandl':   
            # Here get all the required features
            
            features=['Open','Close','High','Low','Volume','Adj. Close']
            
            full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=features,start_dt=app_plots.vars['start_date'],end_dt=app_plots.vars['end_date'])
                        ) for sym in symbols]
            
            
            script_ele,div_ele=plot_symbols_interactive(full_data,'Quandl')
            
            #computed_df=compute_params(full_data)
            
            #script_param,div_param=plot_params(full_data)
            
            return render_template('plot_prices_beta_1.html',script_symbols=script_ele,plot_div_symbols=div_ele
                                  )
        
        elif app_plots.vars['data_src']=='get_data_yahoo': 
            # TODO: This needs a lot of change (pass not yet implemented)
            # data_dict=apicall_data.get_data_from_yahoo(symbol=symbols, features=features)
            return render_template('plot_prices_beta_1.html',error_yahoo='<font size="3" color="red"> Data source not yet enabled </font>')
        
        # Here when the user clicks submit button, the bokeh plot should be displayed


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
    #p=Area(adj_prices_df, width=1200, height=500,tools=[TOOLS,hover])
    p.title.text = "Data for requested %s ticker symbols from %s data source"%(", ".join(list_symbols),data_src)
    p.legend.location = "top_left"
    p.grid.grid_line_alpha=0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price (Adjusted Close)'
    p.ygrid.band_fill_color="olive"
    p.ygrid.band_fill_alpha = 0.1
    
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
    
    
    script_el_data, div_el_data=components(p)
        
    return script_el_data, div_el_data