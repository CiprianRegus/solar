from flask import Flask, jsonify
from flask import g
import sqlite3
import sys
from importlib import reload

sys.path.insert(0, '..')
import main

reload(main)

DATABASE = '../database/database.db'
app = Flask(__name__)

@app.route("/run")
def hello():
    return jsonify({"id": "12", "class": "c"})


@app.route("/load", methods=["GET"])
def load_model():
    main.main("../plant1Data")
    pass


@app.route('/db')
def db_test():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("select * from credentials")
    rows = cursor.fetchall()
    print(rows)


