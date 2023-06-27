import json

class DataSynthesis:
     def __init__(self, variance, examples_to_generate):
         self.variance = variance
         self.examples_to_generate = examples_to_generate
     def data_synthesis(self, optimizer, prompt, response, **kwargs):
        final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'+'\nGenerate '+str(self.examples_to_generate)+' more JSONS each with a prompt and response field like the given one. The content of the prompt and response fields must be similar to the given JSON. Seperate each JSON with a ,.'
        data=optimizer.call_big(final_prompt , kwargs)
        datapoints = []
        data = '{ "datapoints": ['+data+']}'
        print(data)
        data = json.loads(data)
        for i in data["datapoints"]:
            datapoints.append((i["prompt"],i["response"]))
        return datapoints
