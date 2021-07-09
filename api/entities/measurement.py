import sys
import json

sys.path.insert(0, '..')
from init import *

class Measurement(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, nullable=False)
    device_id = db.Column(db.Integer, db.ForeignKey('device.id'), nullable=False)
    device = db.relationship('Device', backref='measurement', lazy=True)
    value = db.Column(db.Float, nullable=False)
    unit_id = db.Column(db.Integer, db.ForeignKey('unit.id'), nullable=False)
    type_id = db.Column(db.Integer, db.ForeignKey('type.id'), nullable=False)
    unit = db.relationship('Unit', backref='measurement', lazy=True)
    measurement_type = db.relationship('Type', backref='measurement', lazy=True)
    predicted = db.Column(db.Boolean, nullable=False)

    def toJSON(self):
        ret = {"date_time": date_time, "value": value}
        return json.dumps(ret)

db.create_all()
