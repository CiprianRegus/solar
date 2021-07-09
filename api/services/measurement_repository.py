import sys
import datetime
from . import device_repository as drep

sys.path.insert(0, '..')
from init import *
import entities.measurement as m
from entities import device as d
from entities import unit as u
from entities import type as t

class MeasurementRepository():

    def __init__(self, db):
        """
            db: SQLAlchemy instance 
        """
        self.db = db
        self.devrep = drep.DeviceRepository(db)
        
    def get_all(self, predicted):
        
        ret = m.Measurement.query.all()
        return ret

    def get_latest(self, predicted, count):
        
        ret = m.Measurement.query.order_by(Measurement.date_time).limit(count).all()
        return ret

    def get_latest_by_device_id(self, predicted, physical_id, value_type, count):
        
        device_id = ""
        try:
            device_id = self.devrep.get_by_id(physical_id) 
        except:
            print("Invalid device_id: ", physical_id)

        tid = 0
        if value_type == "DC":
            tid = 2

        ret = m.Measurement.query.filter_by(device_id=device_id).filter_by(type_id=tid) \
                            .filter_by(predicted=predicted).order_by(m.Measurement.date_time.desc()).limit(count).all()
        return ret

    
    def get_daily_by_device_id(self, predicted, physical_id, value_type, date_time):
        
        tid = 0
        if value_type == "DC":
            tid = 2
 
        device_id = ""
        try:
            device_id = self.devrep.get_by_id(physical_id) 
        except:
            print("Invalid device_id: ", physical_id)

        date = datetime.datetime.strptime(date_time, "%d-%m-%Y")
        ret = m.Measurement.query.filter_by(device_id=device_id).filter(extract('year', m.Measurement.date_time) == date.year,
                                                                        extract('month', m.Measurement.date_time) == date.month,
                                                                        extract('day', m.Measurement.date_time) == date.day).filter_by(predicted=predicted) \
                                                                        .filter_by(type_id=tid).order_by(m.Measurement.date_time.desc()).limit(96).all()
        return ret


    def insert(self, date, time, physical_id, value, unit, value_type, predicted):
        
        date_time = datetime.datetime.strptime(date + " " + str(datetime.timedelta(seconds=time)), '%d-%m-%Y %H:%M:%S')
        
        device_id = ""
        try:
            device_id = self.devrep.get_by_id(physical_id) 
        except:
            print("Invalid device_id: ", physical_id)
        
        tid = 0
        if value_type == "DC":
            tid = 2

        measurement = m.Measurement(date_time=date_time, device_id=device_id, value=value, unit_id=0, type_id=tid, predicted=predicted) 
        db.session.add(measurement)
        id = db.session.commit()
        return id

    def delete():
        pass

