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

sys.path.insert(0, '..')
import main

reload(main)

DATABASE = '../database/database.db'
app = flask_app
### CORS configuration
CORS(app)

@app.route("/run")
def hello():
    return jsonify({"id": "12", "class": "c"})


@app.route("/load", methods=["GET"])
def load_model():
    return jsonify({"predicted_value": main.main("../plant1Data", True)})


@app.route('/db')
def db_test():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("select * from credentials")
    rows = cursor.fetchall()
    print(rows)

@app.route('/authenticate', methods=['GET', 'POST'])
@cross_origin()
def authenticate():
    
    if request.method == "POST":
        print("post")

        # request.form doesn't work for json data ???
        username = request.json["username"]
        password = request.json["password"]
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("select * from credentials where username like '{}' and password like '{}'".format(username, password))
        rows = cursor.fetchall()
        if len(rows) != 0:
            return jsonify({}), 200
        
        return jsonify({}), 404

    return jsonify({}), 400


"""
   start_consuming is a blocking operation 
"""
q_consumer_thread = threading.Thread(target=consume_queue)
q_consumer_thread.start()
