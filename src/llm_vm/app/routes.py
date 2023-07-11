from flask import Blueprint
bp = Blueprint('bp',__name__)

@bp.route('/', methods=['GET'])
def home():
    return '''home'''

@bp.route('/v1/complete', methods=['POST']) 
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
