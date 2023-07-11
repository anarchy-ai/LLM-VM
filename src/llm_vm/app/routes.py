<<<<<<< HEAD:src/llm_vm/app/routes.py
from flask import Blueprint
bp = Blueprint('bp',__name__)

@bp.route('/', methods=['GET'])
=======
import flask
from flask import request, jsonify
import json
import time
import traceback
import importlib
import openai
import os
import hashlib
from src.llm_vm.completion.optimize import LocalOptimizer
from src.llm_vm.agents.REBEL import agent

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
>>>>>>> origin/main:app.py
def home():
    return '''home'''

@bp.route('/v1/complete', methods=['POST']) 
def optimizing_complete():
    rebel_agent = agent.Agent("", [], verbose=1)
    data = json.loads(request.data)
    static_context = data["context"]
    dynamic_prompt = data["prompt"]
    data_synthesis = False
    finetune = False
    use_rebel_agent = False
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
    
    if "tools" in data.keys():
        if type(data["tools"]) != list:
            return {"status":0, "resp":"Wrong data type for tools list"}
        else:
            tools=[]
            for i in data["tools"]:
                temp_tool_dict = {}
                temp_args_dict = {}
                temp_tool_dict.update({"description":i["description"]})
                temp_tool_dict.update({"dynamic_params":i["dynamic_params"]})
                temp_tool_dict.update({"method":i["method"]})
                temp_args_dict.update({"url":i["url"]})
                temp_args_dict.update({"params":{}})
                for j in i["static_params"].keys():
                    temp_args_dict["params"].update({j:i["static_params"][j]})
                for k in i["dynamic_params"].keys():
                    temp_args_dict["params"].update({k:"{"+k+"}"})
                temp_tool_dict.update({"args":temp_args_dict})
                tools.append(temp_tool_dict)
            rebel_agent.set_tools(tools)
            use_rebel_agent = True
   
    try:
        openai.api_key = data["openai_key"]
        os.environ['OPENAI_API_KEY']=openai.api_key
        print(os.getenv("OPENAI_API_KEY"))
    except:
        return  {"status":0, "resp":"Issue with OpenAI key"}
    
    optimizer.openai_key = openai.api_key
    agent.set_api_key(openai.api_key,"OPENAI_API_KEY")
    try:
        if not use_rebel_agent:
            completion = optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
        else:
            completion = rebel_agent.run(static_context+dynamic_prompt,[])[0]
    except Exception as e:
        return {"status":0, "resp": str(e)}
    
    return {"completion":completion, "status": 200}
