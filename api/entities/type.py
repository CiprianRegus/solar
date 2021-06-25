import sys

sys.path.insert(0, "..")
from init import *

class Type(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20), nullable=False)


db.create_all()
