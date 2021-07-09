from flask import Flask, jsonify, request, Blueprint
from flask import g, current_app
import sqlite3
import sys
from importlib import reload
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
from consumer import queue_page, consume_queue
from init import flask_app
import threading
import util
import json
from init import *
import entities.inverse
from services import measurement_repository
from services import device_repository

sys.path.insert(0, '..')
import main

reload(main)

app = flask_app
### CORS configuration
CORS(app)

mrep = measurement_repository.MeasurementRepository(db)

@app.route("/run")
def hello():
    return jsonify({"id": "12", "class": "c"})


@app.route("/load", methods=["GET"])
def load_model():
    mean_std = main.main("../plant1Data", True)
    for e in mean_std:
        device_id, value_type, mean, std = e
        try:
            db.session.add(entities.inverse.NormalizationCoeff(device_id=device_id, value_type=value_type, mean=mean, std=std))
            db.session.commit()
        except Exception as e:
            print("Mean/std pair not inserted: ", e)

    return jsonify({"predicted_value": mean_std})

@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def authenticate():
    
    if request.method == "POST":
        print("post")
        print(request.json)
        # request.form doesn't work for json data ???
        username = request.json["username"]
        password = request.json["password"]
        rows = util.fetch("select * from credentials where username like '{}' and password like '{}'".format(username, password))
        if len(rows) != 0:
            return jsonify({}), 200
        
        return jsonify({}), 404

    return jsonify({}), 400


@app.route('/devices')
def getDevices():
    
    return jsonify({'devices': ['iCRJl6heRkivqQ3', 'ih0vzX44oOqAx2f', 'pkci93gMrogZuBj', 'rGa61gmuvPhdLxV', 'sjndEbLyjtCKgGv', 'uHbuxQJl8lW7ozc',
    'wCURE6d3bPkepu2', 'z9Y9gH1T5YWrNuG', 'zBIq5rxdHJRwDNY', 'zVJPv84UY57bAof', '1BY6WEcLGh8j5v7', '1IF53ai7Xc0U56Y', '3PZuoBAID5Wc2HD',
    '7JYdWkrLSPkdwr4', 'McdE0feGgRqW7Ca', 'VHMLBKoKgIrUVDU', 'WRmjgnKYAwPKWDb', 'ZnxXDlPa8U1GXgE', 'ZoEaEvLYb1n2sOq', 'adLQvlD726eNBSB',
    'bvBOhCH3iADSZry']})

@app.route('/latest', methods=['POST'])
@cross_origin()
def get_latest_values():
    
    if request.method == "POST":
        #date = request.json["date"]
        device_id = request.json["device_id"]
        value_type = request.json["value_type"]
        predicted = request.json["predicted"]
        result = mrep.get_latest_by_device_id(predicted=predicted, physical_id=device_id, value_type=value_type, count=96)
        #print(result) 

        json_result = []
        for e in result[::-1]:
            json_result.append({"date_time": e.date_time, "value": e.value})

        return jsonify({"latest": json_result}), 200

    return jsonify({}), 400

@app.route('/values_by_day', methods=['POST'])
@cross_origin()
def get_daily_values():
    
    if request.method == "POST":
        date = request.json["date"]
        device_id = request.json["device_id"]
        value_type = request.json["value_type"]
        predicted = request.json["predicted"]
        
        print("Date: ", date)

        result = mrep.get_daily_by_device_id(predicted=predicted, physical_id=device_id, value_type=value_type, date_time=date)

        json_result = []
        for e in result[::-1]:
            json_result.append({"date_time": e.date_time, "value": e.value})

        return jsonify({"latest": json_result}), 200

    return jsonify({}), 400




class User(db.Model):
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username


#user1 = User(username="User1", password="pass123")
#db.session.add(user1)
#db.session.commit()
#print(User.query.all())

"""
   start_consuming is a blocking operation 
"""
q_consumer_thread = threading.Thread(target=consume_queue)
q_consumer_thread.start()

