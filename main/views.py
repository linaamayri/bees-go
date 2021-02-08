from flask import Blueprint, request, url_for, redirect, render_template, flash, jsonify
from datetime import date
from datetime import datetime
import pandas as pd
import pathlib
main = Blueprint('main', __name__, template_folder="templates")

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/about")
def about():
    return "All about Flask"

@main.route("/api/v1/info")
def get_info():
    res = read_data()
    if res == None: 
        return jsonify()
    return jsonify(
        temperature = int(res['temperature']),
        weight = int(res['weight'])
    ) 

def read_data(): 
    date = datetime.now().strftime("%d/%m/%Y")
    data = pd.read_csv(str(pathlib.Path(__file__).parent.absolute()) + "/data1.csv", sep=";")
    for i in range(len(data.date)):
        if data.date[i] == date:
            return {'temperature': data.tempbas[i], 'weight': data.poids[i]}
