from flask_rabmq import RabbitMQ
from flask import Flask, jsonify, request, Blueprint
import pika
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
from init import flask_app, db
import sys
import json
from importlib import reload
import services.measurement_repository as m_rep
import entities.inverse as nval

sys.path.insert(0, '..')
import main
import model_parameter

reload(main)

QUEUE_HOST = '172.17.0.2'

queue_page = Blueprint('queue_page', 'queue_p')
app = flask_app
CORS(app)
### Websockets configuration
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

#Initialize the object responsible for db insertion
measurement_repo = m_rep.MeasurementRepository(db)

def consume_queue():
    conn = pika.BlockingConnection(pika.ConnectionParameters(host=QUEUE_HOST))
    channel = conn.channel()
    channel.queue_declare(queue='TestQueue')

    def callback(ch, method, properties, body):
        received_json = json.loads(body)
        print(received_json['Device_id'])
        
        date = received_json['Date']
        device_id = received_json['Device_id']
        time = float(received_json['Time'])
        temp = float(received_json['Ambiental_temperature'])
        irradiation = float(received_json['Irradiation'])
        prev_ac = float(received_json['Previous_day_ac'])
        prev_dc = float(received_json['Previous_day_dc'])
        ac = float(received_json['Ac_power'])
        dc = float(received_json['Dc_power'])

        dnorm_ac = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="AC_POWER").first()
        dnorm_dc = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="DC_POWER").first()
        temp_coeff = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="AMBIENT_TEMPERATURE").first()
        irradiation_coeff = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="IRRADIATION").first()
        prev_day_ac_coeff = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="PREVIOUS_DAY_AC").first()
        prev_day_dc_coeff = nval.NormalizationCoeff.query.filter_by(device_id=device_id, value_type="PREVIOUS_DAY_DC").first()

        ac_model_input = [model_parameter.ModelParameter(name="Time", value=time),
                      model_parameter.ModelParameter(name="Ambiental_temperature", value=temp, mean=temp_coeff.mean, std=temp_coeff.std), 
                      model_parameter.ModelParameter(name="Irradiation", value=irradiation, mean=irradiation_coeff.mean, std=irradiation_coeff.std),
                      model_parameter.ModelParameter(name="Previous_day_ac", value=prev_ac, mean=prev_day_ac_coeff.mean, std=prev_day_ac_coeff.std)]
        dc_model_input = [model_parameter.ModelParameter(name="Time", value=time),
                      model_parameter.ModelParameter(name="Ambiental_temperature", value=temp, mean=temp_coeff.mean, std=temp_coeff.std), 
                      model_parameter.ModelParameter(name="Irradiation", value=irradiation, mean=irradiation_coeff.mean, std=irradiation_coeff.std),
                      model_parameter.ModelParameter(name="Previous_day_dc", value=prev_ac, mean=prev_day_dc_coeff.mean, std=prev_day_dc_coeff.std)]


        ac_result = main.predict_with_user_input(ac_model_input, dnorm_ac.mean, dnorm_ac.std, "AC")
        dc_result = main.predict_with_user_input(dc_model_input, dnorm_dc.mean, dnorm_dc.std, "DC")
        print(dc_result)
        # Send the prediction result to frontend
        socketio.emit('AC_predict', 
                        {'Time': time,
                        'Date': date,
                        'Device_id': device_id,
                        'Value': ac_result}, namespace='/test')
        socketio.emit('DC_predict', 
                        {'Time': time,
                        'Date': date,
                        'Device_id': device_id,
                        'Value': dc_result}, namespace='/test')

        socketio.emit('AC', 
                        {'Time': time,
                        'Date': date,
                        'Device_id': device_id,
                        'Value': ac}, namespace='/test')
        
        socketio.emit('DC', 
                        {'Time': time,
                        'Date': date,
                        'Device_id': device_id,
                        'Value': dc}, namespace='/test')

        #Store in database
        measurement_repo.insert(date, time, device_id, ac_result, "KW", "AC", True)
        measurement_repo.insert(date, time, device_id, dc_result, "KW", "DC", True)
        measurement_repo.insert(date, time, device_id, ac, "KW", "AC", False)
        measurement_repo.insert(date, time, device_id, dc, "KW", "DC", False)

    channel.basic_consume(queue='TestQueue', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

@socketio.on('request', namespace='/test')
def test_sock(data):
    print("Request received: ", data)
    socketio.emit('response', 
        {'data': 'Test data'}, namespace='/test')

@app.route('/io')
def send_sock_message():
    
    socketio.emit("response", "Message", namespace='/test')
    return jsonify({}), 200
