import sys

sys.path.insert(0, '..')
from init import db

class Inverse(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    value_type = db.Column(db.String(10), nullable=False)
    device_id = db.Column(db.String(20), nullable=False)
    value = db.Column(db.Float, nullable=False)

class NormalizationCoeff(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    value_type = db.Column(db.String(10), nullable=False)
    device_id = db.Column(db.String(20), nullable=False)
    mean = db.Column(db.Float, nullable=False)
    std = db.Column(db.Float, nullable=False)

db.create_all()
