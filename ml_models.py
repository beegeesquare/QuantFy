from flask import Flask,render_template,redirect,request, Blueprint, flash, url_for
import os
from ediblepickle import checkpoint


from collections import defaultdict

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

        #app_mlModels.vars['window']=int(request.form['window']) # Default window size is set to 20
        app_mlModels.vars['ml_algo']= request.form['ml_algo']
        app_mlModels.vars['future_days']=int(request.form['future_days'])
        #print request.form
        
        full_data=[(sym, apicall_data.get_data_from_quandl(symbol=sym, features=['Adj. Close'], start_dt=app_mlModels.vars['start_date'],
                                                           end_dt=app_mlModels.vars['end_date'])
                        ) for sym in app_mlModels.vars['sym']]
        
        # Convert this to required format
        prices_df=util.get_data(full_data)
        
        # Drop the bench mark price
        prices_df=prices_df.drop(bench_sym,axis=1)
        
        metrics_df,future_df=computeFeatures(prices_df,app_mlModels.vars['future_days']) # Prices should technically have one symbol, but also possible that they might have multiple symbols
        
        # Plot the time-series graph with past and future values
        
        script_el_data, div_el_data=plotPredicted(prices_df, future_df)
        
        return render_template('ml_models.html', script_el_data=script_el_data,div_el_data=div_el_data,metrics=metrics_df.to_html())
        

def computeFeatures(df,shift=5,window=20):
    '''
    This function computes the features that are needed for the machine learning model
    '''
   
    df=df.fillna(df.mean()) # Fill the NaN values with mean
    
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
    # Metrics dataframe
    
    metrics_df=pd.DataFrame(index=['Mean square error','Correlation coff'],columns=df.columns)   
    future_start_date=df.index[-1]; # Take the last date + 1 as the future first date
    future_start_date=future_start_date+dt.timedelta(days=1)
    
    future_end_date=future_start_date+dt.timedelta(days=shift-1); # -1 because we have already included the first date
    future_df=pd.DataFrame(columns=df.columns,index=pd.date_range(future_start_date,future_end_date,freq='D'))
    
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
                       
        # print X_data.tail(),Y_data.tail()
        # For each symbol we need to create a seprate model ()
        #bestEst=buildEstimator(X_data,Y_data,app_mlModels.vars['ml_algo'])
        
        #featureData[sym]=(X_data,Y_data,bestEst)
        
        # Predict the Y for the last sample point, which will give the price for the shift days
        
        #Y_pred_dr=bestEst.predict(X_data.ix[-2*shift:-shift])
        #print Y_pred_dr, Y_data.ix[-2*shift:-shift]
        #print mean_squared_error(Y_data.ix[-2*shift:-shift], Y_pred_dr)
        #print np.corrcoef(np.array(Y_pred_dr),np.array(Y_data.ix[-2*shift:-shift])) # I need to reshape Y_pred 
        
        ######
        # bestEst=buildEstimator(X_data.ix[:-shift],Y_data.ix[:-shift],app_mlModels.vars['ml_algo'])
        scaler=StandardScaler()
        Y_scaled=scaler.fit_transform(df[sym].ix[:-shift])
        
        bestEst=buildEstimator(X_data.ix[:-shift],Y_scaled,app_mlModels.vars['ml_algo'])
        
        Y_pred_dr=bestEst.predict(X_data.ix[-2*shift:-shift]) # This is for computing showing the metrics
        
        # Do the inverse-transform
        Y_pred_dr=scaler.inverse_transform(Y_pred_dr)
        Y_pred_dr= np.reshape(Y_pred_dr,(len(Y_pred_dr),))
        
        tmp_df=np.array(df[sym].ix[-shift:])
        
        tmp_df=np.reshape(tmp_df,(tmp_df.shape[0],))
        
        mse= mean_squared_error(tmp_df, Y_pred_dr)
        
        correlation_mat= np.corrcoef(Y_pred_dr,tmp_df)
        
        correction_coff=correlation_mat[0,1]; # This is a symmetrical matrix of size 2 x 2
        
        metrics_df.loc['Mean square error',sym]=mse
        metrics_df.loc['Correlation coff',sym]=correction_coff
        
        # Prediction for next "shift" days
        
        Y_future=bestEst.predict(X_data.ix[-shift:])
        # Inverse transform
        Y_future=scaler.inverse_transform(Y_future)
        
        future_df[sym]=np.transpose(Y_future)
    
    # print  metrics_df
    # print future_df
    return  metrics_df,  future_df    


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
    
def plotPredicted(df,future_df):
    
    script_el_data=''
    div_el_data=''
    
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
    p.title.text = "Data for requested %s ticker symbols"%(", ".join(list_symbols))
    
       
    colors=viridis(len(list_symbols))
    for (i,sym) in enumerate(list_symbols):
        #p.line(df.index[-20:],df[sym].ix[-20:],line_width=2,legend=sym,color=colors[i]) 
        p.line(df.index,df[sym],line_width=2,legend=sym,color=colors[i]) 
        p.line(future_df.index,future_df[sym],line_width=2,legend=sym,color=colors[i],line_dash='dashed')
        
    future_start_line = Span(location=time.mktime(df.index[-1].timetuple())*1000,
                              dimension='height', line_color='red',
                              line_dash='dashed', line_width=3)  
    past_text=Label(x=time.mktime(df.index[-20].timetuple())*1000,y=df.ix[-1,:].sum()/2,text='Past')
    future_text=Label(x=time.mktime(future_df.index[0].timetuple())*1000,y=future_df.ix[0,:].sum()/2,text='Future')
    p.add_layout(future_start_line)
    p.add_layout(past_text) 
    p.add_layout(future_text)
    script_el_data, div_el_data=components(p)
    
    return script_el_data, div_el_data 