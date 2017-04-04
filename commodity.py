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
import apicall_data

app_commodity=Flask(__name__)





commodity = Blueprint('commodity_prices.html', __name__, template_folder='templates')
@commodity.route('/commodity',methods=['GET','POST'])
def commodity_models():
    if request.method == 'GET':
        return render_template('commodity_prices.html')
    
