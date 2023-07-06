import json
import sys 

class DataSynthesis:
     def __init__(self, variance, examples_to_generate):
         self.variance = variance
         self.examples_to_generate = examples_to_generate
     def data_synthesis(self, optimizer, prompt, response,example_delim="<Datum-Separator/>", **kwargs):
        final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'+ \
            '\nGenerate '+str(self.examples_to_generate)+F""" more JSONS each with a prompt and response field like the given one. 
            The content of the prompt and response fields must be similar to the given JSON. 
            Separate each JSON with the XML tag {example_delim}."""
        data = None       
        response=optimizer.call_big(final_prompt , kwargs)
        datapoints = []
        print(response,file=sys.stderr)
        split_response = response.split(sep=example_delim)
        datum_failure = 0 
        bad_key_failure =0
        resp_filter = {}
        for d in split_response:
            try: 
                the_data = json.loads(d)
                the_tuple = (the_data["prompt"],the_data["response"])
                if the_tuple in resp_filter:
                    continue   # dont save a response if its already happened
                resp_filter[the_tuple]=True  # for now we're treating the (Q,A) pair as a single value
                datapoints.append(the_tuple)
            except json.decoder.JSONDecodeError as err: 
                print(F'data_synthesis response parsing failed with: { err }',file=sys.stderr)
                datum_failure+=1
            except LookupError as err : # i have no evidence that this will happen
                print(F'data_synthesis key lookup failed with: { err }',file=sys.stderr)
                bad_key_failure +=1
        print(F'Out of { len(split_response)} response objects, {datum_failure} were not valid json \n\
            and {bad_key_failure} were missing a key',file=sys.stderr)
        return datapoints
