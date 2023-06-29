import json

class DataSynthesis:
     def __init__(self, variance, examples_to_generate):
         self.variance = variance
         self.examples_to_generate = examples_to_generate
     def data_synthesis(self, optimizer, prompt, response,retries = 1 , **kwargs):
        final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'+ \
            '\nGenerate '+str(self.examples_to_generate)+' more JSONS each with a prompt and response field like the given one. The content of the prompt and response fields must be similar to the given JSON. Separate each JSON with a ,.'
        data = None       
        attempt_count = 0       
        # the little statemachine here should be replaced with looping on some 
        # enum about the job state, but this does the right thing for now. TODO
        while(data == None and attempt_count <= retries ):
            response=optimizer.call_big(final_prompt , kwargs)
            datapoints = []
            attempt_count = 0 
            data_str = '{ "datapoints": ['+response+']}'
            print(data_str)
            try: 
                data = json.loads(data_str)
                break # exit the loop 
            except json.decoder.JSONDecodeError as err: 
                print(F'data_synthesis response parsing failed with: { err }')
                continue # retry data synthesis
                
        for i in data["datapoints"]:
            datapoints.append((i["prompt"],i["response"]))
        return datapoints
