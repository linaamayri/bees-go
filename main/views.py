from flask import Blueprint, request, url_for, redirect, render_template, flash, jsonify
from datetime import date
from datetime import datetime
import pandas as pd
import pathlib
main = Blueprint('main', __name__, template_folder="templates")

@main.route("/")
def index():
    return render_template("index.html")

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

@main.route("/api/v1/air")
def get_air():
    postal_code = int(request.args.get('postalCode'))
    date = datetime.strptime(request.args.get('date'), "%Y-%m-%d")
    res = read_air_data(postal_code, date)
    return jsonify(res)

def read_air_data(postal_code, date):
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/air.csv", sep=",")
    res = []
    for i in range(len(data.ninsee)):
        if data.ninsee[i] == postal_code and pd.to_datetime(data.date[i], format='%d/%m/%Y') == date:
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
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/data1.csv", sep=";")
    for i in range(len(data.date)):
        if data.date[i] == date:
            return {'temperature': data.tempbas[i], 'weight': data.poids[i]}
