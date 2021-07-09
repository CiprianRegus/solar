import sys

sys.path.insert(0, '..')
from init import *
import entities.measurement as m
from entities import device as d
from entities import unit as u
from entities import type as t
  
class DeviceRepository():
 
    def __init__(self, db):
        """
            db: SQLAlchemy instance 
        """
        self.db = db


    def get_by_id(self, device_id):
        
        ret = d.Device.query.filter_by(physical_id=device_id).first().id
        return ret

    def insert(self, device_id, type):

        device = d.Device(physical_id=device_id, type=type)
        db.session.add(device)
        db.session.commit()
