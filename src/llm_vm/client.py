import openai
from llm_vm.agents.REBEL import agent
from llm_vm.completion.optimize import LocalOptimizer
import llm_vm.completion.models as models

class Client:
    def __init__(self, big_model = 'gpt', small_model ='chat_gpt'):
        # load optimizer for endpoint use
        MODELCONFIG = models.ModelConfig(big_model=big_model, small_model=small_model)
        print("Using model: " + big_model)
        def CALL_BIG(prompt, **kwargs):
            model = MODELCONFIG.big_model
            return model.generate(prompt,**kwargs)

        def CALL_SMALL(prompt,**kwargs):
            model = MODELCONFIG.small_model
            return model.generate(prompt,**kwargs)

        self.optimizer = LocalOptimizer(MIN_TRAIN_EXS=2,openai_key=None, call_big=CALL_BIG, call_small= CALL_SMALL)

    
    def complete(self, prompt,
                 context,
                 openai_key = "",
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
        # if openai_key == "":
        #     return {"status":0, "resp":"No OpenAI key provided"}
    
        kwargs.update({"temperature":temperature})

        if stoptoken is not None:
            kwargs.update({"stop":stoptoken})
            
        if tools is not None:
            use_rebel_agent = True

        try:
            openai.api_key = openai_key
        except:
            return  {"status":0, "resp":"Issue with OpenAI key"}
    
        self.optimizer.openai_key = openai.api_key
        agent.set_api_key(openai.api_key,"OPENAI_API_KEY")

        try:
            if not use_rebel_agent:
                completion = self.optimizer.complete(static_context,dynamic_prompt,data_synthesis=data_synthesis,finetune = finetune, **kwargs)
            else:
                completion = rebel_agent.run(static_context+dynamic_prompt,[])[0]
        except Exception as e:
            return {"status":0, "resp": str(e)}
        return {"completion":completion, "status": 200}

