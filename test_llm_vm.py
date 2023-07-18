# import llm_vm.client as l 
import os, requests, json

openai_key = os.getenv('OPENAI_API_KEY')

url = "http://localhost:3002/v1/complete"
json_payload = {"prompt": "what is the economic situation in canada?",
                "context": "",
                "temperature": 0.0,
                "openai_key": openai_key,
                # "finetune": True,
                }
                
response = requests.post(url, data=json.dumps(json_payload))
print(response.text)
