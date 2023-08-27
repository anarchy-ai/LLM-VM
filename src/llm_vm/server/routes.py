import sys
from flask import request, Blueprint
import json
import os
import openai
from llm_vm.agents.REBEL import agent
from llm_vm.client import Client
from llm_vm.config import settings
# load optimizer for endpoint use
# optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None)

client = Client( big_model=settings.big_model, small_model=settings.small_model)

print('optimizer loaded', file=sys.stderr)

bp = Blueprint('bp',__name__)

@bp.route('/', methods=['GET'])
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

    if "stoptoken" in data.keys():

        if type(data["stoptoken"]) != str and type(data["stoptoken"]) != list:
            # stop can either be a string or array of strings
            return {"status":0, "resp":"Wrong Data Type for stop"}
        elif type(data["stoptoken"]) == list:
            if len(data["stoptoken"]) > 4:
                return {"status":0, "resp":"Too many stop tokens in array limit to 4 or less"}
            # check that every element in the list is a string
            for j in data["stoptoken"]:
                if type(j) != str:
                    return {"status":0, "resp":"Wrong Data Type for stop"}
            kwargs.update({"stop": data["stoptoken"]})
        else:
            kwargs.update({"stop":data["stoptoken"]})

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
    except:
        return  {"status":0, "resp":"Issue with OpenAI key"}

    # optimizer.openai_key = openai.api_key
    agent.set_api_key(openai.api_key,"OPENAI_API_KEY")
    try:
        if not use_rebel_agent:
            completion = client.complete(static_context,dynamic_prompt,openai_key=openai.api_key, data_synthesis=data_synthesis,finetune = finetune, **kwargs)
        else:
            completion = rebel_agent.run(static_context+dynamic_prompt,[])[0]
    except Exception as e:
        return {"status":0, "resp": str(e)}

    # return {"completion":completion, "status": 200}
    return completion
