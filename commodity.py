from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for

import numpy as np
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
            app_commodity.vars['start_date']=dt.datetime.today()-dt.timedelta(days=30*365) # This does not give the accurate 10 years
        
        
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
        
        
        return render_template('commodity_prices.html')