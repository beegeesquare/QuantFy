from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for
import os
from ediblepickle import checkpoint


from collections import defaultdict

import numpy as np
from bokeh.plotting import figure, show,output_file,ColumnDataSource
from bokeh.palettes import viridis
from bokeh.embed import components
from bokeh.layouts import gridplot, WidgetBox

from bokeh.models import HoverTool, Span, Label
from bokeh.charts import Area
from bokeh.models.widgets import Panel, Tabs, DataTable,TableColumn
#from bokeh.io import vform


import time
import math
import pandas as pd
import datetime as dt
import util
import apicall_data

import statsmodels.api as sm

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,make_scorer

dict_algo_names_map={"mean_algo":'Mean-model',"knn_algo":'k-Nearest neighbours',"rf_algo":'Random Forest Regressor',
                     "lr_algo":'Linear Regression',"ridge_algo":'Ridge CV'} # This should same as in ml_models.html. Otherwise, it will given an error

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

        #app_mlModels.vars['window']=int(request.form['window']) # Default window size is set to 20
        app_mlModels.vars['ml_algo']= request.form['ml_algo']
        app_mlModels.vars['bench_ml_algo']=request.form['bench_ml_algo']
        
        if app_mlModels.vars['ml_algo']==app_mlModels.vars['bench_ml_algo']:
            return render_template('ml_models.html',error_same_mdl='<font size="3" color="red" > ML-algorithm and the Benchmark should be different </font>')
        
        app_mlModels.vars['future_days']=int(request.form['future_days'])
        #print request.form
        
        full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=['Adj. Close'], start_dt=app_mlModels.vars['start_date'],
                                                           end_dt=app_mlModels.vars['end_date'])
                        ) for sym in app_mlModels.vars['sym']]
        
        # Convert this to required format
        prices_df=util.get_data(full_data)
        
        # Drop the bench mark price
        prices_df=prices_df.drop(bench_sym,axis=1)
        split_number=100; # This can be changed as needed, primarly use for spliting the data into train (train+cv) and test set
        
        metrics_df,future_df,bench_future_df,pct_tab=computeFeatures(prices_df,app_mlModels.vars['future_days'],split_number) # Prices should technically have one symbol, but also possible that they might have multiple symbols
        
        
        # print metrics_df    
        # Plot the time-series graph with past and future values
        
        tab1=plotPredicted(prices_df, future_df,dict_algo_names_map[app_mlModels.vars['ml_algo']],split_number)
        
        tab2=plotPredicted(prices_df,bench_future_df, dict_algo_names_map[app_mlModels.vars['bench_ml_algo']],split_number)
        
        tabs=Tabs(tabs=[tab1,tab2,pct_tab])
        
        script_el_data,div_el_data=components(tabs)
        # Put the metrics table in the html using bokeh
        metrics_data=dict(metrics_df[[i for i in metrics_df.columns]].round(4) )
        #print metrics_data
        metrics_data['Metric']=metrics_df.index # This will add the index (Note: Instead of Metric, if I use index, then the width of output index column cannot be adjustested )
        source=ColumnDataSource(metrics_data)
        columns=[TableColumn(field=i,title=i) for i in metrics_df.columns]
        # Insert the index column to the list of columns
        columns.insert(0, TableColumn(field="Metric",title="Metric"))
        metrics_table=DataTable(source=source,columns=columns, height=250, width=400)
        script_metrics,div_metrics= components(WidgetBox(metrics_table))
        
        # Similarly for the price describe data 
        price_describe=prices_df.describe().round(4)
        price_describe_data=dict(price_describe[[i for i in price_describe]] )
        price_describe_data['Metric']=price_describe.index # This will add the index (Note: Instead of Metric, if I use index, then the width of output index column cannot be adjustested )
        source=ColumnDataSource(price_describe_data)
        columns=[TableColumn(field=i,title=i) for i in price_describe.columns]
        # Insert the index column to the list of columns
        columns.insert(0, TableColumn(field="Metric",title="Metric"))
        price_describe_table=DataTable(source=source,columns=columns,height=250,width=400)
        script_price_describe,div_price_describe= components(WidgetBox(price_describe_table))
        
        return render_template('ml_models.html', script_el_data=script_el_data,div_el_data=div_el_data,
                                               script_metrics=script_metrics,div_metrics=div_metrics,
                                               script_price_describe=script_price_describe,div_price_describe=div_price_describe)
        

def computeFeatures(df,shift=5,split_number=60):
    '''
    This function computes the features that are needed for the machine learning model
    Shift is the future-days and all the rolling means. volatility etc has to be computed accordingly
    '''
    
    df=df.fillna(df.mean()) # Fill the NaN values with mean
    
    # Compute the rolling-mean and rolling-std
    rolling_mean=util.get_rolling_mean(df, shift)
    rolling_std=util.get_rolling_std(df,shift)
    
    # Find compute the normalized Bollinger bands (which is different from the one in the util function)
    normalized_bb=util.compute_normalized_BB(df, rolling_mean, rolling_std, shift)
            
    # Compute standard-deviation of daily-return, which is nothing but volatility
    volatility=util.compute_volatility(df, shift)
    
    
    # Compute momentum feature
    momentum=util.compute_momentum(df, shift)
    
    # All the above computed features could contain multiple symbols
    featureData=defaultdict()
    # Metrics dataframe
    
    metrics_df=pd.DataFrame(index=['MSE','Correlation coff'],columns=df.columns)   
    future_start_date=df.index[-1]; # Take the last date + 1 as the future first date
    future_start_date=future_start_date+dt.timedelta(days=1)
    
    future_end_date=future_start_date+dt.timedelta(days=shift-1); # -1 because we have already included the first date
    future_df=pd.DataFrame(columns=df.columns,index=pd.date_range(future_start_date,future_end_date,freq='D'))
    bench_future_df=future_df.copy(); # Create a copy for benchmark predictions
    
    # These will be used for error plots
    full_Y_shifted=df.shift(-shift)
    
    full_Y_test=full_Y_shifted.ix[:-shift,:].ix[-split_number:,:]
    
    full_Y_pred=pd.DataFrame(index=full_Y_test.index) 
    full_Y_pred_bench=full_Y_pred.copy() # This is for benchmark algorithm
        
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
        
                
        # Create train and test-set
        # split_number=60; # This is the number of days that we want to leave for test set; kind of cut-off date
        X_pred=X_data.ix[-shift:,:] # For this the values Y will be NaN and we will use that to predict the future stocks
        
        X_train=X_data.ix[:-shift,:].ix[:-split_number,:]; # Take the train set from the data that is left out for prediction
        X_test=X_data.ix[:-shift,:].ix[-split_number:,:] # This will be used to get the prediction
      
        Y_shifted=df[sym].shift(-shift); # The last values in this are NaN
        
        Y_pred=Y_shifted.ix[-shift:]; # These values are NaN which needs to be predicted
        # print Y_pred
        Y_train=Y_shifted.ix[:-shift].ix[:-split_number]
        Y_test=Y_shifted.ix[:-shift].ix[-split_number:]
        
        
        ######
        # bestEst=buildEstimator(X_data.ix[:-shift],Y_data.ix[:-shift],app_mlModels.vars['ml_algo'])
        scaler=StandardScaler()
        
        
        Y_train_scaled=scaler.fit_transform(Y_train)
        
        bestEst=buildEstimator(X_train,Y_train_scaled,app_mlModels.vars['ml_algo'])
        
        # print app_mlModels.vars['bench_ml_algo']
        
        bench_bestEst=buildEstimator(X_train,Y_train_scaled,app_mlModels.vars['bench_ml_algo'])
        
        # Now predict Y using the test set (this test set has not been seen by the mode)
        # Given algo
        Y_pred_test=bestEst.predict(X_test) # This is for computing showing the metrics
        # benchmark prediction
        bench_Y_pred_test=bench_bestEst.predict(X_test)
        
        # Do the inverse-transform
        Y_pred_inverse=scaler.inverse_transform(Y_pred_test)
        Y_pred_test= np.reshape(Y_pred_inverse,(len(Y_pred_inverse),))
        
        full_Y_pred[sym]=pd.Series(Y_pred_test,index=Y_test.index)
        
        
        # Inverse-transform for benchmark algorithm
        bench_Y_pred_inverse=scaler.inverse_transform(bench_Y_pred_test)
        bench_Y_pred_test=np.reshape(bench_Y_pred_inverse,(len(bench_Y_pred_inverse),))
        full_Y_pred_bench[sym]=pd.Series(bench_Y_pred_test,index=Y_test.index) # Index must be passed here
        
        
        tmp_df=np.array(Y_test)
        tmp_df=np.reshape(tmp_df,(tmp_df.shape[0],))
       
        
        
        
        # For the given algorithm
        mse= mean_squared_error(tmp_df, Y_pred_test)
        correlation_mat= np.corrcoef(Y_pred_test,tmp_df)
        correction_coff=correlation_mat[0,1]; # This is a symmetrical matrix of size n x n; n=number of ticker symbols
        
        # For the benchmark algorithm
        bench_mse= mean_squared_error(tmp_df, bench_Y_pred_test)
        bench_correlation_mat= np.corrcoef(bench_Y_pred_test,tmp_df)
        bench_correction_coff=bench_correlation_mat[0,1]; # This is a symmetrical matrix of size n x n
        #print bestEst,bench_bestEst
        #print mse,correction_coff, bench_mse,bench_correction_coff
        
        metrics_df.loc['MSE',sym]=mse
        if math.isnan(correction_coff)==False:
            metrics_df.loc['Correlation coff',sym]=correction_coff
        # Add the benchmark metrics
        
        metrics_df.loc['Benchmark MSE',sym]=bench_mse
        if math.isnan(bench_correction_coff)==False:
            metrics_df.loc['Benchmark Correlation coff',sym]=bench_correction_coff
       
        
        #metrics_df=metrics_df.round(4)
        # Prediction for next "shift" days
        
        Y_future=bestEst.predict(X_pred)
        bench_Y_future=bench_bestEst.predict(X_pred)
        # Inverse transform
        Y_future=scaler.inverse_transform(Y_future)
        bench_Y_future=scaler.inverse_transform(bench_Y_future)
        
        future_df[sym]=np.transpose(Y_future)
        bench_future_df[sym]=np.transpose(bench_Y_future)
        
    # print  metrics_df
    # print future_df
    # Plot the percentage errors
    
    pct_plot_tab=percentage_error_plot(full_Y_test,full_Y_pred,full_Y_pred_bench)
    # Drop the NA values, especially with Mean model correlation coeffecient
    metrics_df=metrics_df.dropna(axis=0).round(4)
    # print metrics_df
    return  metrics_df,  future_df, bench_future_df,pct_plot_tab


def buildEstimator(X_data,Y_data,ml_model):
    """
    Given the dataset of the symbol (historical), it will return the estimator
    """
    tscv=TimeSeriesSplit(n_splits=5) # Creates a time-series cross-validation set (creates 3 fold train and test sets, roll forward CV)
    scorer=make_scorer(mean_squared_error,greater_is_better=False)
    if ml_model=="knn_algo":
        
        param_grid={'n_neighbors':range(3,20,1)}
        
        n_neighbours_cv=GridSearchCV(KNeighborsRegressor(),param_grid=param_grid,cv=tscv,scoring=scorer)
        
        n_neighbours_cv.fit(X_data,Y_data); # It splits and creates CV sets
        
        #print n_neighbours_cv.grid_scores_
        #print n_neighbours_cv.cv_results_
        #print n_neighbours_cv.best_estimator_

        # Return the optimal model
        bestEst= n_neighbours_cv.best_estimator_
                   
        return bestEst
    
    if ml_model=="rf_algo":
        
        #print ml_model
        param_grid={'oob_score':[True,False]}
        rfr_cv=GridSearchCV(RandomForestRegressor(),param_grid=param_grid,cv=tscv,scoring=scorer)
        
        rfr_cv.fit(X_data,Y_data)
        
        bestEst=rfr_cv.best_estimator_
        
        return bestEst
    
    if ml_model=="lr_algo":
        
        param_grid={'n_estimators':range(10,100,5)}
        
        lr_cv=GridSearchCV(LinearRegression(),param_grid=param_grid,cv=tscv,scoring=scorer)
        
        lr_cv.fit(X_data,Y_data)
        
        bestEst=lr_cv.best_estimator_
        
        # print bestEst
        
        return bestEst
    
       
    if ml_model=="mean_algo":
        # This simply returns the mean value of the stock prices
        # print ml_model
        mean_reg=DummyRegressor(strategy='mean')
        mean_reg.fit(X_data,Y_data)
        # print mean_reg.get_params()
        # print mean_reg
        
        return mean_reg
    
    if ml_model=="ridge_algo":
        param_grid={'fit_intercept':['True','False'],'solver':['auto','svd','lsqr','sag'],'alpha': np.logspace(-4., 0, 20)}
        
        ridge_cv=GridSearchCV(Ridge(),param_grid=param_grid,cv=tscv,scoring=scorer)
        
        ridge_cv.fit(X_data,Y_data)
        
        bestEst=ridge_cv.best_estimator_
        
        # print bestEst
        
        return bestEst
    
    
def plotPredicted(df,future_df,algo,split_number=60):
    
    # Just draw one plot for all the ticker symbols requested for
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
    
    hover=HoverTool(
            tooltips=[
                ("Adj. Close",'$y'),
                
            ]
        )
    
    list_symbols=list(df.columns)
    
    p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price (Adjusted Close)'
    
    p.title.text = "Prediction graph for %s ticker symbols using %s"%(", ".join(list_symbols),algo)
    
       
    colors=viridis(len(list_symbols))
    for (i,sym) in enumerate(list_symbols):
        p.line(df.index[-split_number:],df[sym].ix[-split_number:],line_width=2,legend=sym,color=colors[i]) 
        #p.line(df.index,df[sym],line_width=2,legend=sym,color=colors[i]) 
        p.line(future_df.index,future_df[sym],line_width=2,legend=sym,color=colors[i],line_dash='dashed')
        
    future_start_line = Span(location=time.mktime(df.index[-1].timetuple())*1000,
                              dimension='height', line_color='red',
                              line_dash='dashed', line_width=3)  
    past_text=Label(x=time.mktime(df.index[-20].timetuple())*1000,y=df.ix[-1,:].sum()/2,text='Past')
    future_text=Label(x=time.mktime(future_df.index[0].timetuple())*1000,y=future_df.ix[0,:].sum()/2,text='Future')
    p.add_layout(future_start_line)
    p.add_layout(past_text) 
    p.add_layout(future_text)
    
    tab=Panel(child=p,title=algo)
    
    return tab 

def percentage_error_plot(y_true,y_pred,y_bench):
    
    """
    Given the true value and the predicted value, it computes the 
    """
    TOOLS='pan,wheel_zoom,box_zoom,reset,save,box_select,crosshair'
    
    hover=HoverTool(
            tooltips=[
                ("Error pct",'$y'),
                
            ]
        )
    
    list_symbols=list(y_true.columns)
    
    p = figure(width=900, height=500, x_axis_type="datetime",tools=[TOOLS,hover])
    p.legend.location = "top_left"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Error percentage'
    
    p.title.text = "Error percentage for %s ticker symbols"%(", ".join(list_symbols))
    
    
       
    colors=viridis(len(list_symbols))
    for (i,sym) in enumerate(list_symbols):
        p.line(y_true.index,(y_true[sym]-y_pred[sym])/y_true[sym]*100,legend=sym,line_width=2,color=colors[i]) 
        
        #p.line(y_true.index,(y_true[sym]-y_bench[sym])/y_true[sym]*100,line_width=2,legend='Bench_'+sym,color=colors[i],line_dash='dashed')
    
    
    tab=Panel(child=p,title='Error percentage')
    
    return tab

