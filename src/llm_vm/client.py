
import openai
from llm_vm.agents.REBEL import agent
from llm_vm.completion.optimize import LocalOptimizer

# load optimizer for endpoint use
optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None)

def complete(prompt,
             context,
             openai_key,
             finetune=False,
             data_synthesis = False,
             temperature=0,
             stoptoken = None,
             tools = None):
    
    rebel_agent = agent.Agent("", [], verbose=1)
    static_context = context
    dynamic_prompt = prompt
    use_rebel_agent = False
    kwargs = {}
    if openai_key is None:
        return {"status":0, "resp":"No OpenAI key provided"}
        
    
    kwargs.update({"temperature":temperature})

    if stoptoken is not None:
        kwargs.update({"stop":stoptoken})                 
        
    

    
    if tools is not None:
        use_rebel_agent = True

    try:
        openai.api_key = openai_key
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


    
    
