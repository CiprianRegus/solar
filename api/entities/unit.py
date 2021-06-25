import sys

sys.path.insert(0, '..')
from init import *

class Unit(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    unit = db.Column(db.String(10), nullable=False)


db.create_all()
