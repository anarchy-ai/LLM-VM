import json
import sys
from sentence_transformers import SentenceTransformer, util
import openai
import os
import time
import pickle
import llm_vm.config as conf
from llm_vm.guided_completion import RegexCompletion, ChoicesCompletion, TypeCompletion, GrammarCompletion
from transformers import AutoTokenizer

class DataSynthesis:
    def __init__(self, variance, examples_to_generate):
        self.variance = variance
        self.examples_to_generate = examples_to_generate
    def data_synthesis(self, optimizer, prompt, response, openai_key=None, regex = None, type = None, choices = None, **kwargs):
        """
        This method generates QA pairs using the larger LLM to be used as training data for fine-tuning the smaller LLM.

        Parameters
        ----------
        - optimizer (class): The Optimizer class to use for fine-tuning. Could be either LocalOptimizer or HostedOptimizer.
        - prompt (str | list ): A question to be used as a one-shot QA example for the larger LLM prompt.
        - response (str | list): A verified answer to the provided prompt question to be used in the one-shot QA example.
        - example_delim (str): A unique XML tag used to separate the generated JSON examples. Default value is "<Datum-Separator/>".
        - semantic_sim (bool): Option to use semantic similarity to filter duplicate generations. Default value is True.
        - **kwargs: Additional keyword arguments to be passed into the `call_big` method.

        Returns
        ----------
        - List: A list of tuples containing the QA pairs to be used for fine-tuning.
        """
        
        if os.path.isfile(conf.settings.data_gen_file):
            new_file = open(conf.settings.data_gen_file,"rb")
            return list(pickle.load(new_file))
        datapoints = []
        final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'
        final_prompt = "Generate 1 json similar to the one below. \n" + final_prompt
        
        while len(datapoints) < self.examples_to_generate:
            datapoint = self.generate_example(final_prompt, openai_key, regex = regex, type = type, choices = choices)
            time.sleep(5)
            datapoints.append(datapoint)
            print(datapoint)
        new_file = open(conf.settings.data_gen_file,"wb")
        pickle.dump(datapoints,new_file)
        return datapoints
    
    def generate_example(self, final_prompt, openai_key, example_delim = "<END>", model="gpt-4",max_tokens = 1000,temperature = 1,regex = None,type = None,choices = None, grammar_type = None):
        openai.api_key=openai_key
        cur_prompt = [{'role': "system", 'content' : final_prompt}]

        response = openai.ChatCompletion.create(messages=cur_prompt,model=model,max_tokens=max_tokens,temperature=temperature)['choices'][0]['message']['content']
        
        if regex is not None:
            try:
                the_data = json.loads(response.replace("\n",""))
                prompt = the_data["prompt"]
                response = RegexCompletion.complete(prompt,regex)
                the_tuple = (prompt,response+example_delim)
            except:
                pass
            
        elif type is not None:
            try:
                the_data = json.loads(response.replace("\n",""))
                prompt = the_data["prompt"]
                response = TypeCompletion.complete(prompt,type)
                the_tuple = (prompt,response+example_delim)
            except:
                pass

        elif choices is not None:
            try:
                the_data = json.loads(response.replace("\n",""))
                prompt = the_data["prompt"]
                response = ChoicesCompletion.complete(prompt,choices)
                the_tuple = (prompt,response+example_delim)
            except:
                pass

        elif grammar_type is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", padding_side='left')
                constraint_model = GrammarCompletion("gpt2-medium", tokenizer)
                the_data = json.loads(response.replace("\n",""))
                prompt = the_data["prompt"]
                response = constraint_model.complete(prompt, grammar_type=grammar_type)
                the_tuple = (prompt,response+example_delim)
            except:
                pass


        else:
            try:
                the_data = json.loads(response.replace("\n",""))
                the_tuple = (the_data["prompt"],the_data["response"]+example_delim)
            except:
                pass
        
        return the_tuple