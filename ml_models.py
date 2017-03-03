from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for
import os
from ediblepickle import checkpoint


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

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import mean_squared_error,make_scorer


app_mlModels=Flask(__name__)

mlModels = Blueprint('ml_models.html', __name__, template_folder='templates')
@mlModels.route('/mlModels',methods=['GET','POST'])
def getMLmodels():
    if request.method=='GET':
        return render_template('ml_models.html')
    else:
        # check if the post request has the file part
        app_mlModels.vars={}
        app_mlModels.vars['ml_algo']=request.form['ml_algo']
        
        bench_sym='SPY'; # This is not available for the user 
        
        if (request.form['sym']=='') :  # sym is a list delimited by ;
            return render_template('ml_models.html',error_sym='<font size="3" color="red" > Provide at least one ticker symbol </font>') 
        else:
            app_mlModels.vars['sym']=request.form['sym'].upper().strip(';').split(';')
            
            app_mlModels.vars['sym'].insert(0,'SPY'); # SPY is the bench mark symbol which is not accessible to the user
            
            
        if len(request.form['start_date'])!=0: # Here start and end date are keys are coming even if they are empty
            try:
                app_mlModels.vars['start_date']=dt.datetime.strptime(request.form['start_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('ml_models.html',error_start_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take 5 years ago of the current date
            app_mlModels.vars['start_date']=dt.datetime.today()-dt.timedelta(days=5*365) # This does not give the accurate 5 years
        
        
        if  len(request.form['end_date'])!=0:
            try:
                app_mlModels.vars['end_date']=dt.datetime.strptime(request.form['end_date'],'%m/%d/%Y')
            except ValueError:
                return render_template('ml_models.html',error_end_date='<font size="3" color="red" > Wrong date format </font>')
        else:
            # Take today as the default date
            app_mlModels.vars['end_date']=dt.datetime.today()

        app_mlModels.vars['window']=int(request.form['window'])
        app_mlModels.vars['ml_algo']= request.form['ml_algo']
        app_mlModels.vars['future_days']=int(request.form['future_days'])
        print request.form
        
        full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=['Adj. Close'], start_dt=app_mlModels.vars['start_date'],
                                                           end_dt=app_mlModels.vars['end_date'])
                        ) for sym in app_mlModels.vars['sym']]
        
        # Convert this to required format
        prices_df=util.get_data(full_data)
        
        # Drop the bench mark price
        prices_df=prices_df.drop(bench_sym,axis=1)
        
        computeFeatures(prices_df,app_mlModels.vars['window'],app_mlModels.vars['future_days']) # Prices should technically have one symbol, but also possible that they might have multiple symbols
        
        return render_template('ml_models.html')
        

def computeFeatures(df,window=20,shift=5):
    '''
    This function computes the features that are needed for the machine learning model
    '''
   
    
    
    # Compute the rolling-mean and rolling-std
    rolling_mean=util.get_rolling_mean(df, window)
    rolling_std=util.get_rolling_std(df,window)
    
    # Find compute the normalized Bollinger bands (which is different from the one in the util function)
    normalized_bb=util.compute_normalized_BB(df, rolling_mean, rolling_std, window)
            
    # Compute standard-deviation of daily-return, which is nothing but volatility
    volatility=util.compute_volatility(df, window)
    
    
    # Compute momentum feature
    momentum=util.compute_momentum(df, window)
    
    # All the above computed features could contain multiple symbols
    featureData=defaultdict()
    # Create train and test sets for each symbol
    for sym in df.columns:
        X_data=pd.DataFrame(index=df.index)
        
        # Join all the features
        X_data=X_data.join(normalized_bb[sym])
        X_data=X_data.rename(columns={sym:'Norm_BB'})
        #
        X_data=X_data.join(volatility[sym])
        X_data=X_data.rename(columns={sym:'Volatility'})
        #
        X_data=X_data.join(momentum[sym])
        X_data=X_data.rename(columns={sym:'Momentum'})
        # y values X_datare the shifted return computed as ret = (price[t+10]/price[t]) - 1.0
        
        Y_data=pd.DataFrame(index=df.index,columns=[sym])
        # This is the return based on shifted value, basically if shift is 1, then it is daily return
        
        Y_data[sym].ix[:-shift] = np.array((df[sym].ix[shift:]/df[sym].ix[:-shift].values) - 1); # Here np is choosen, because if dataframe was used then Y_values will not be shifted
        Y_data[sym].ix[-shift:]=0
                       
        print X_data.tail(),Y_data.tail()
        # For each symbol we need to create a seprate model ()
        bestEst=buildEstimator(X_data,Y_data,app_mlModels.vars['ml_algo'])
        
        featureData[sym]=(X_data,Y_data,bestEst)
        
        # Predict the Y for the last sample point, which will give the price for the shift days
        
        Y_pred_dr=bestEst.predict(X_data.ix[-2*shift:-shift])
        print Y_pred_dr, Y_data.ix[-2*shift:-shift]
        print mean_squared_error(Y_data.ix[-2*shift:-shift], Y_pred_dr)
        #print np.corrcoef(np.array(Y_pred_dr),np.array(Y_data.ix[-2*shift:-shift])) # I need to reshape Y_pred 
        
        ######
        bestEst=buildEstimator(X_data.ix[:-shift],Y_data.ix[:-shift],app_mlModels.vars['ml_algo'])
        
        Y_pred_dr=bestEst.predict(X_data.ix[-shift:])
        print Y_pred_dr
        print mean_squared_error(Y_data.ix[-shift:], Y_pred_dr)
        
    return       


def buildEstimator(X_data,Y_data,ml_model):
    """
    Given the dataset of the symbol (historical), it will return the estimator
    """
    if ml_model=="knn_algo":
        tscv=TimeSeriesSplit(n_splits=3) # Creates a time-series cross-validation set (creates 3 fold train and test sets)
        param_grid={'n_neighbors':range(3,15,1)}
        scorer=make_scorer(mean_squared_error,greater_is_better=False)
        n_neighbours_cv=GridSearchCV(KNeighborsRegressor(),param_grid=param_grid,cv=tscv,scoring=scorer)
        
        n_neighbours_cv.fit(X_data,Y_data); # It splits and creates CV sets
        
        print n_neighbours_cv.cv_results_
        print n_neighbours_cv.best_estimator_

        # Return the optimal model
        bestEst= n_neighbours_cv.best_estimator_
        
        
    return bestEst