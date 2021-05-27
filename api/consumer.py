from flask_rabmq import RabbitMQ
from flask import Flask, jsonify, request, Blueprint
import pika
from flask_socketio import SocketIO
from flask_cors import CORS, cross_origin
from init import flask_app
import sys
import json
from importlib import reload

sys.path.insert(0, '..')
import main

reload(main)

queue_page = Blueprint('queue_page', 'queue_p')
app = flask_app
CORS(app)
### Websockets configuration
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

def consume_queue():
    conn = pika.BlockingConnection(pika.ConnectionParameters(host='172.17.0.2'))
    channel = conn.channel()
    channel.queue_declare(queue='TestQueue')

    def callback(ch, method, properties, body):
        received_json = json.loads(body)
        print(received_json['Device_id'])
        model_input = [float(received_json['Time']), float(received_json['Ambiental_temperature']), float(received_json['Irradiation']),
                        float(received_json['Previous_day_ac'])]
        result = main.predict_with_user_input(model_input)
        # Send the prediction result to frontend
        socketio.emit('result', 
                        {'Time': received_json['Time'],
                        'Value': result}, namespace='/test')

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
