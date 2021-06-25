import sys

sys.path.insert (0, '..')
from init import *

class Measured_value(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    measurement_id = db.Column(db.Integer, db.ForeignKey('measurement.id'))
    measurements = db.relationship('Measurement', lazy=True)

