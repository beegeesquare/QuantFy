from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for

import numpy as np
import statsmodels.api as sm
import sklearn.linear_model

from bokeh.plotting import figure, show,output_file,ColumnDataSource
from bokeh.palettes import viridis
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, Span, Label
from bokeh.charts import Area


import time
import pandas as pd
import datetime as dt
import util
from apicall_data import get_commodity_data_from_quandl

app_commodity = Flask(__name__)





commodity = Blueprint('commodity_prices.html', __name__, template_folder='templates')
@commodity.route('/commodity',methods=['GET','POST'])
def commodity_models():
    if request.method == 'GET':
        return render_template('commodity_prices.html')
    else:
        # check if the post request has the file part
        app_commodity.vars = {}
        app_commodity.vars['data_code'] = request.form['data_code']
        app_commodity.vars['comm_code'] = request.form['comm_code']
        
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_commodity.vars['start_date'] = dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('commodity_prices.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 5 years ago of the current date
            app_commodity.vars['start_date']=dt.datetime.today()-dt.timedelta(days=30*365) # This does not give the accurate 30 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_commodity.vars['end_date'] = dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('commodity_prices.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_commodity.vars['end_date'] = dt.datetime.today()
        
        comm_df = get_commodity_data_from_quandl(app_commodity.vars['data_code'], 
                                                 app_commodity.vars['comm_code'],
                                                 app_commodity.vars['start_date'],
                                                 app_commodity.vars['end_date'])
        # the dataframe consists of column name as 'Value', change it to Price
        comm_df=comm_df.rename(columns={'Value':'Price'})
        
        exp_model_df=long_term_estimation(comm_df)
        
        script_el_data, div_el_data=plot_commodity_interactive(exp_model_df)
        
        return render_template('commodity_prices.html', script_el_data=script_el_data, div_el_data=div_el_data)

def long_term_estimation(comm_df):
    """
    This function takes the basic value plot, and provides the long term estimator 
    based on the exponential model
    """
    comm_df['Julian'] = comm_df.index.to_julian_date() # Converts the index into Julian float value
    comm_df = sm.add_constant(comm_df) # Add a constant field for the linear regression
    # We can actually train a simple exponential model using the log(value), then train further models on the error.
    exponential_model = sklearn.linear_model.Ridge().fit( 
        X=comm_df[['Julian', 'const']], 
        y=np.log(comm_df['Price'])
    )
    
    exp_model_df = comm_df
    exp_model_df['Exponential_Model'] = np.exp(exponential_model.predict(comm_df[['Julian', 'const']]))
    exp_model_df['Log_Error_Exponential'] = np.log(comm_df['Price'] / comm_df['Exponential_Model'])
    
    return exp_model_df


def plot_commodity_interactive(comm_df):
    """
    Input: Takes the commodity data and commputes the 
    This function returns the div and script element of the commodity plot.
    
    """
    usr_price_features = ['Price','Estimator']
    # Just draw one plot for all the ticker symbols requested for
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
    
    hover=HoverTool(
            tooltips=[
                ("Price",'$y')
            ]
        )
    
    p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
  
    # Draw the plot for the oil price
    p.line(comm_df.index, comm_df.Price, color='blue',line_width=2,legend='Price')
    p.line(comm_df.index, comm_df['Exponential_Model'], color='red',line_width=2,legend='Exponential_Model')
    
    p.title.text = "Data for requested %s commodity from %s data source"%(app_commodity.vars['comm_code'],
                                                                               app_commodity.vars['data_code'])
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    
    script_el_data, div_el_data=components(p)
        
    return script_el_data, div_el_data
