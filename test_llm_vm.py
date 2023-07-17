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

json_payload = {"prompt": "what is the economic situation in england?",
                "context": "",
                "temperature": 0.0,
                "openai_key": openai_key,
                # "finetune": True,
                }
response = requests.post(url, data=json.dumps(json_payload))
print(response.text)

json_payload = {"prompt": "what is the economic situation in france?",
                "context": "",
                "temperature": 0.0,
                "openai_key": openai_key,
                "finetune": True,
                }
response = requests.post(url, data=json.dumps(json_payload))
print(response.text)
# ret = l.complete("Who are the top 10 greatest soccer players", "", openai_key)
# assert type(ret) == str
# ret = l.complete("Who are the top 10 greatest soccer players", "", "")
# assert type(ret) == Exception
# ret = l.complete("What should lions do if they are hungry", "", openai_key, finetune=0)
# assert type(ret) == Exception
# ret = l.complete("What should lions do if they are hungry", "", openai_key, finetune=True, data_synthesis= 90)
# assert type(ret) == Exception
# ret = l.complete("", "", openai_key, finetune=True, data_synthesis= 90)
# assert type(ret) == Exception
# ret = l.complete("What is 2+2+2", "Answer the following math problem with just a number", openai_key, finetune=True)
# assert type(ret) == str
# assert ret.strip() == "6"
# print("All tests passed")
