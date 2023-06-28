import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
from src.llm_vm.completion.optimize import LocalOptimizer

# from test_agent import run_test
from flask_cors import CORS

from contextlib import contextmanager

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True



@app.route('/', methods=['GET'])
def home():
    return '''home'''

@app.route('/complete', methods=['POST']) 
def optimizing_complete():
    data = json.loads(request.data)
    static_context = data["context"]
    dynamic_prompt = data["prompt"]
    data_synthesis = False
    finetune = False
    kwargs = {}
    if "temperature" in data.keys():
        if type(data["temperature"]) != float or type(data["temperature"]) != int:
            raise Exception("Wrong Data Type for temperature")
        else:
            kwargs.update({"temperature":data["temperature"]})
    
    if "data_synthesis" in data.keys():
        if type(data["data_synthesis"])==bool:
            data_synthesis = data["data_synthesis"]
        else:
            raise Exception("Wrong Data Type for data_synthesis")
    
    if "finetune" in data.keys():
        if type(data["finetune"])==bool:
            finetune = data["finetune"]
        else:
            raise Exception("Wrong Data Type for finetune")
    
    openai.api_key = data["openai_key"]
    os.environ['OPENAI_API_KEY']=openai.api_key
    print(os.getenv("OPENAI_API_KEY"))
    #try:
        
    # except:
    #     raise Exception("No OpenAI Key Provided")
    
    optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=openai.api_key)
    completion = optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)

    return {"completion":completion, "status": 200}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3002)