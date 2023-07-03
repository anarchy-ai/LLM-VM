import app
import requests

app.run(host="0.0.0.0", port=3002)
class llm_vm:
    def complete(self, prompt, context, finetune=False, data_synthesis = False, temperature=0):
        
        url = '0.0.0.0:3002/complete'
        params = {'prompt': prompt, "context":context, "finetune":finetune}
        if temperature != 0:
            params.update({"temperature":temperature})

        x = requests.post(url, json = params)
        return x
    
    def get_data(self, context, temperature=0):
        
        url = '0.0.0.0:3002/complete'
        params = {'context': context}
        if temperature != 0:
            params.update({"temperature":temperature})

        x = requests.post(url, json = params)
        return x

