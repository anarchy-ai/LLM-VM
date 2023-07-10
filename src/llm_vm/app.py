import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib
from llm_vm.completion.optimize import LocalOptimizer


# from test_agent import run_test
from flask_cors import CORS

from contextlib import contextmanager

optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None)
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

def generate_hash(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(input_string).encode('utf-8'))
    return int(sha256_hash.hexdigest(), 16) % 10**18

@app.route('/', methods=['GET'])
def home():
    return '''home'''

@app.route('/v1/complete', methods=['POST']) 
def optimizing_complete():
    data = json.loads(request.data)
    static_context = data["context"]
    dynamic_prompt = data["prompt"]
    data_synthesis = False
    finetune = False
    kwargs = {}
    if "openai_key" not in data.keys():
        return {"status":0, "resp":"No OpenAI key provided"}
        
    if "temperature" in data.keys():
        if type(data["temperature"]) != float and type(data["temperature"]) != int:
            return {"status":0, "resp":"Wrong Data Type for temperature"}
        else:
            kwargs.update({"temperature":data["temperature"]})
    
    if "data_synthesis" in data.keys():
        if type(data["data_synthesis"])==bool:
            data_synthesis = data["data_synthesis"]
        else:
            return {"status":0, "resp":"Wrong Data Type for data_synthesis"}
    
    if "finetune" in data.keys():
        if type(data["finetune"])==bool:
            finetune = data["finetune"]
        else:
            return {"status":0, "resp":"Wrong Data Type for finetune"}
    
   
    try:
        openai.api_key = data["openai_key"]
        os.environ['OPENAI_API_KEY']=openai.api_key
        print(os.getenv("OPENAI_API_KEY"))
    except:
        raise Exception("No OpenAI Key Provided")
    
    optimizer.openai_key = openai.api_key
    try:
        completion = optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
    except:
        return {"status":0, "resp":"An error occured"}
    
    return {"completion":completion, "status": 200}


def main_server_entry_point():
    # make this more configurable soon
    app.run(host="192.168.1.75", port=3002)

if __name__ == '__main__':
    app.run(host="192.168.1.75", port=3002)