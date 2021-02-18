from __future__ import print_function
import sys
from flask import Blueprint, request, url_for, redirect, render_template, flash, jsonify, current_app, session 
from datetime import date
from datetime import datetime
import pandas as pd
import pathlib
import numpy as np
import pickle


#simple_geoip = SimpleGeoIP(current_app)


# Read the data from the file
main = Blueprint('main', __name__, template_folder="templates")
#with open(str(pathlib.Path(__file__).parent.absolute())+ '/model/pickled_departments.pkl', 'rb') as fid:
#    departments = pickle.load(fid)
#with open(str(pathlib.Path(__file__).parent.absolute())+ '/model/pickled_x_test.pkl', 'rb') as fid:
#    x_test = pickle.load(fid)
#with open(str(pathlib.Path(__file__).parent.absolute())+ '/model/pickled_model.pkl', 'rb') as fid:
#    model = pickle.load(fid)
#with open(str(pathlib.Path(__file__).parent.absolute())+ '/model/pickled_scal.pkl', 'rb') as fid:
#    scaler = pickle.load(fid)
    

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

# create future forecast dates
def create_dates(start,days):
    v = pd.date_range(start=start, periods=days+1, freq='D', closed='right')
    number_days_forecast = pd.DataFrame(index=v) 
    return number_days_forecast


@main.route("/")
def index():
    return render_template("version.html")

@main.route("/installation")
def installation():
    return render_template("installation.html")

@main.route("/predateurs")
def predateurs():
    return render_template("predateurs.html")

@main.route("/recolte")
def recolte():
    return render_template("recolte.html")

@main.route("/about")
def about():
    return "All about Flask"

@main.route("/api/v1/weight")
def get_weight():
    date_from = datetime.strptime(request.args.get('dateFrom'), "%Y-%m-%d")
    date_to = datetime.strptime(request.args.get('dateTo'), "%Y-%m-%d")
    res = read_weight_data(date_from, date_to)
    return jsonify(res)

def read_weight_data(date_from, date_to):
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/data/weight.csv", sep=";")
    res = []
    for i in range(len(data.date)):
        date = pd.to_datetime(data.date[i], format='%d/%m/%Y')
        if date >= date_from and date < date_to :
            res.append({
                "date": str(data.date[i]),
                "value": int(data.value[i])
            })
    return res
    


@main.route("/api/v1/air")
def get_air():
    postal_code = int(request.args.get('postalCode'))
    date_from = datetime.strptime(request.args.get('dateFrom'), "%Y-%m-%d")
    date_to = datetime.strptime(request.args.get('dateTo'), "%Y-%m-%d")
    res = read_air_data(postal_code, date_from, date_to)
    return jsonify(res)

def read_air_data(postal_code, date_from, date_to):
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/data/air.csv", sep=",")
    res = []
    for i in range(len(data.ninsee)):
        date = pd.to_datetime(data.date[i], format='%d/%m/%Y')
        if data.ninsee[i] == postal_code and date >= date_from and date < date_to :
            res.append({
                "date": str(data.date[i]),
                "postalCode": int(data.ninsee[i]),
                "no2": int(data.no2[i]),
                "o3": int(data.o3[i]),
                "pm10": int(data.pm10[i])
            })
    return res


@main.route("/api/v1/info")
def get_info():
    res = read_temperature_data()
    if res == None: 
        return jsonify()
    return jsonify(
        temperature = int(res['temperature']),
        weight = int(res['weight'])
    ) 

def read_temperature_data(): 
    date = datetime.now().strftime("%d/%m/%Y")
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/data/temperature.csv", sep=";")
    for i in range(len(data.date)):
        if data.date[i] == date:
            return {'temperature': data.tempbas[i], 'weight': data.poids[i]}


@main.route("/ee")
def ee():
    seven_day_forecast_svr = create_dates('2021-01-31',31) 
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], 31, model[i], 60, scaler[i])            
        seven_day_forecast_svr[departments[i]] = np.array(seven_days_svr) 
    print(seven_day_forecast_svr)
    return render_template("index.html")

@main.route("/prevision")
def prevision():
    #session['sequence'] = 0
    #seg = session.get('sequence')
    #simple_geoip = SimpleGeoIP(current_app)
    #geoip_data = simple_geoip.get_geoip_data()

    depart = 'Val-de-Marne'
    currentdate = datetime.today().strftime('%Y-%m-%d')
    last = datetime.strptime('2021-01-31', "%Y-%m-%d")
    d2 = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), "%Y-%m-%d")
    days_current = abs((d2 - last).days)

    current_forecast = create_dates('2021-01-31',days_current) 
    demain_forcast = create_dates('2021-01-31',(days_current+1))
    seven_forcats = create_dates('2021-01-31',(days_current+7))
    quinze_forcats = create_dates('2021-01-31',(days_current+15))
    tente_forcats = create_dates('2021-01-31',(days_current+30))
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], days_current, model[i], 60, scaler[i])            
        current_forecast[departments[i]] = np.array(seven_days_svr)
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], days_current+1, model[i], 60, scaler[i])            
        demain_forcast[departments[i]] = np.array(seven_days_svr) 
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], days_current+7, model[i], 60, scaler[i])            
        seven_forcats[departments[i]] = np.array(seven_days_svr) 
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], days_current+15, model[i], 60, scaler[i])            
        quinze_forcats[departments[i]] = np.array(seven_days_svr) 
    for i in range(len(departments)):
        seven_days_svr = forecast_svr(x_test[i], days_current+30, model[i], 60, scaler[i])            
        tente_forcats[departments[i]] = np.array(seven_days_svr)
    tente_forcats = tente_forcats.tail(30)
    quinze_forcats = quinze_forcats.tail(15)
    seven_forcats = seven_forcats.tail(7)
    demain_forcast = demain_forcast.tail(1)
    current_forecast = current_forecast.tail(1)
    print(seven_forcats.tail(7))
    return render_template("/prevision/homePrevision.html", prevision=current_forecast)