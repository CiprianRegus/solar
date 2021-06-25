import sys

sys.path.insert(0, '..')
from init import *

class Device(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    physical_id = db.Column(db.String(20), nullable=False, unique=True)
    type = db.Column(db.String(30), nullable=False)
    devices = db.relationship('Measurement', backref='device', lazy=True)


db.create_all()
