# Import the require library
from flask import Flask, request, jsonify
import redis
from celery import Celery


# we are Initialize Flask app
app = Flask(__name__)

# Initialize Redis connection
redis_conn = redis.StrictRedis(host='localhost',port=6379, db=0)

# Initialize Celery
celery = Celery(app.name, broker='redis://localhost:6379/0')

# Example REST endpoint with queuing
@app.route('/process_data', methods=['POST'])
# define the function
def process_data():
    data = request.get_json()

    # Add data to the Redis queue
    redis_conn.rpush('data_queue', str(data))
    # Then Trigger the Celery task to process the data asynchronously
    process_data_task.delay(data)
    # Then, we are returning the message
    return jsonify({'message':'Data added to the queue for processing'})

# Celery task to process data asynchronously
@celery.task
# define the function
def process_data_task(data):
    # data processing logic here
    # Example: print the data
    print(f"Processing data: {data}")

# define the entry point for the program
if __name__ == "__main__":
    app.run(debug=True)