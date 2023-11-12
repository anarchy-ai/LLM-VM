# Import the require library
from kombu import Exchange, Queue

broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

task_queues = (
    Queue('default', Exchange('default'), routing_key='default'),
)

task_routes = {
    'your_module.process_data_task': {'queue': 'default'},
}