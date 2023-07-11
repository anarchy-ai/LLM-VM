
import requests
import json

class llm_vm:
    def complete(prompt, context, openai_key, finetune=False, data_synthesis = False, temperature=0):
        
        url = 'http://192.168.1.75:3002/v1/complete'
        params = {'prompt': prompt, "context":context, "finetune":finetune, "data_synthesis":data_synthesis, "openai_key":openai_key, "temperature":temperature}
        x = requests.post(url, json = params)
        if json.loads(x.text)["status"] == 0:
            return Exception(json.loads(x.text)["resp"])
    
        return json.loads(x.text)["completion"]
    
    
