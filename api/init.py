from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import extract

flask_app = Flask(__name__)
flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../database/database.db'
db = SQLAlchemy(flask_app)

db.create_all()
