import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import math
from datetime import datetime as dt
from IPython.display import Image, HTML

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_table('model/temperature-quotidienne-departementale.txt')
df1 = pd.pivot_table(df, values = 'tmoy', index=['date_obs'], columns = 'departement').reset_index()

# Change to datetime format
df_temp = df1.copy() 
df_temp[df_temp.columns[0]] = pd.to_datetime(df_temp[df_temp.columns[0]], format='%d/%m/%Y')
df_temp = df_temp.set_index(df_temp.columns[0])
df_temp = df_temp.sort_index()

# create future forecast dates
def create_dates(start,days):
    v = pd.date_range(start=start, periods=days+1, freq='D', closed='right')
    number_days_forecast = pd.DataFrame(index=v) 
    return number_days_forecast

# get department and it's temperature,  and drop null values
def get_temperatue_department(df_temp,i):
    temp_value = df_temp[[df_temp.columns[i]]].dropna()
    department_name = df_temp.columns[i]
    return temp_value, department_name 

# train-test split for a user input ratio
def train_test_split(value, name, ratio):
    nrow = len(value)
    print(name+' total samples: ',nrow)
    split_row = int((nrow)*ratio)
    print('Training samples: ',split_row)
    print('Testing samples: ',nrow-split_row)
    train = value.iloc[:split_row]
    test = value.iloc[split_row:]
    return train, test, split_row 

# data transformation
def data_transformation(train_tract1,test_tract1):
    scaler = MinMaxScaler()
    train_tract1_scaled = scaler.fit_transform(train_tract1)
    test_tract1_scaled = scaler.fit_transform(test_tract1)          
    train_tract1_scaled_df = pd.DataFrame(train_tract1_scaled, index = train_tract1.index, columns=[train_tract1.columns[0]])
    test_tract1_scaled_df = pd.DataFrame(test_tract1_scaled,
                                         index = test_tract1.index, columns=[test_tract1.columns[0]])
    return train_tract1_scaled_df, test_tract1_scaled_df, scaler 

    # feature builder - This section creates feature set with lag number of predictors--Creating features using lagged data
def timeseries_feature_builder(df, lag):
    df_copy = df.copy()
    for i in range(1,lag):
        df_copy['lag'+str(i)] = df.shift(i) 
    return df_copy
    df_copy = df.copy()

    # preprocessing -- drop null values and make arrays 
def make_arrays(train_tract1,test_tract1):
    X_train_tract1_array = train_tract1.dropna().drop(train_tract1.columns[0], axis=1).values
    y_train_tract1_array = train_tract1.dropna()[train_tract1.columns[0]].values
    X_test_tract1_array = test_tract1.dropna().drop(test_tract1.columns[0], axis=1).values
    y_test_tract1_array = test_tract1.dropna()[test_tract1.columns[0]].values    
    return X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array

# fitting & Validating using SVR
def fit_svr(X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array):
    model_svr = SVR(kernel='rbf', gamma='auto', tol=0.001, C=10.0, epsilon=0.001)
    model_svr.fit(X_train_tract1_array,y_train_tract1_array)
    y_pred_train_tract1 = model_svr.predict(X_train_tract1_array)
    y_pred_test_tract1 = model_svr.predict(X_test_tract1_array)        
    print('r-square_SVR_Test: ', round(model_svr.score(X_test_tract1_array,y_test_tract1_array),2))
    return model_svr, y_pred_test_tract1  

# validation result  
def valid_result_svr(scaler, y_pred_test_tract1, station_value, split_row, lag):
    new_test_tract1 = station_value.iloc[split_row:]
    test_tract1_pred = new_test_tract1.iloc[lag:].copy()
    y_pred_test_tract1_transformed = scaler.inverse_transform([y_pred_test_tract1])
    y_pred_test_tract1_transformed_reshaped = np.reshape(y_pred_test_tract1_transformed,(y_pred_test_tract1_transformed.shape[1],-1))
    test_tract1_pred['Forecast'] = np.array(y_pred_test_tract1_transformed_reshaped)
    return test_tract1_pred

# multi-step future forecast
def forecast_svr(X_test_tract1_array, days ,model_svr, lag, scaler):
    last_test_sample = X_test_tract1_array[-1]        
    X_last_test_sample = np.reshape(last_test_sample,(-1,X_test_tract1_array.shape[1]))        
    y_pred_last_sample = model_svr.predict(X_last_test_sample)                
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample
    new_array = X_last_test_sample
    new_predict = y_pred_last_sample

    seven_days_svr=[]
    for i in range(0,days):               
            new_array = np.insert(new_array, 0, new_predict)                
            new_array = np.delete(new_array, -1)
            new_array_reshape = np.reshape(new_array, (-1,lag))                
            new_predict = model_svr.predict(new_array_reshape)
            temp_predict = scaler.inverse_transform([new_predict])
            seven_days_svr.append(temp_predict[0][0].round(2))
            
    return seven_days_svr 

def france_temp_svr2(df_temp, lag):     
    #seven_day_forecast_svr = create_dates('2021-01-31',days) 
    departments = []
    x_test = []
    model = []
    scal =  []
    for i in range(len(df_temp.columns)):
        # preprocessing
        #station_value, station_name = get_value_name(all_station_temp,i)
        temp_value, department_name = get_temperatue_department(df_temp,i)         
        train_tract1, test_tract1, split_row = train_test_split(temp_value, department_name, 0.85)              
        train_tract1_scaled_df, test_tract1_scaled_df, scaler = data_transformation(train_tract1,test_tract1)        
        train_tract1 = timeseries_feature_builder(train_tract1_scaled_df,lag+1)
        test_tract1 = timeseries_feature_builder(test_tract1_scaled_df, lag+1)        
        X_train_tract1_array, y_train_tract1_array, X_test_tract1_array, y_test_tract1_array = make_arrays(train_tract1,
                                                                                                           test_tract1)
        # SVR modeling
        model_svr, y_pred_test_tract1 = fit_svr(X_train_tract1_array, y_train_tract1_array,
                                                X_test_tract1_array, y_test_tract1_array)                       
        test_tract1_pred = valid_result_svr(scaler, y_pred_test_tract1, temp_value, split_row, lag)  

        departments.append(department_name)
        x_test.append(X_test_tract1_array)
        model.append(model_svr)
        scal.append(scaler)

        #seven_days_svr = forecast_svr(X_test_tract1_array, days, model_svr, lag, scaler)            
        #seven_day_forecast_svr[department_name] = np.array(seven_days_svr)        
    return departments, x_test, model, scal

departments, x_test, model, scal = france_temp_svr2(df_temp,60)



with open('model/pickled_departments.pkl', 'wb') as fid:
     pickle.dump(departments, fid)
with open('model/pickled_x_test.pkl', 'wb') as fid:
     pickle.dump(x_test, fid)
with open('model/pickled_model.pkl', 'wb') as fid:
     pickle.dump(model, fid)
with open('model/pickled_scal.pkl', 'wb') as fid:
     pickle.dump(scal, fid)


