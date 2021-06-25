import sys
import datetime
import time

sys.path.insert(0, "..")
from init import *

class Record(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, nullable=False)
    ambient_temperature = db.Column(db.Float, nullable=False)
    irradiation = db.Column(db.Float, nullable=False)
    previous_day_dc = db.Column(db.Float, nullable=False)
    previous_day_ac = db.Column(db.Float, nullable=False)
   

db.create_all()
r = Record(date_time=datetime.datetime.now(), ambient_temperature=22.4, irradiation=0.754, previous_day_dc=1.75, previous_day_ac=2.21)
db.session.add(r)
db.session.commit()
rec = Record.query.all()

print(rec)
