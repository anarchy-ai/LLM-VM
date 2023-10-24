import json
import sys
from sentence_transformers import SentenceTransformer, util
import openai
import backoff 
import os
import time
import pickle
import llm_vm.config as conf
from llm_vm.guided_completion import GenerativeCompletion

class DataSynthesis:
    def __init__(self, variance, examples_to_generate):
        self.variance = variance
        self.examples_to_generate = examples_to_generate

    def data_synthesis(self, optimizer, prompt, response, openai_key=None, completion=None, **kwargs):
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
        self.optimizer = optimizer
        if os.path.isfile(conf.settings.data_gen_file):
            new_file = open(conf.settings.data_gen_file,"rb")
            return list(pickle.load(new_file))
        datapoints = []
        final_prompt = '{"prompt": "' + prompt +'"  , "response": "' +response+'" }'
        final_prompt = "Generate {self.examples_to_generate} json similar to the one below and separate each json using a new line. \n" + final_prompt
        datapoints = self.generate_example(final_prompt, openai_key, completion=completion)
        if len(datapoints) > self.examples_to_generate:
            datapoints = datapoints[:self.examples_to_generate]
        new_file = open(conf.settings.data_gen_file,"wb")
        pickle.dump(datapoints,new_file)
        return datapoints
    
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def generate_example(self, final_prompt, openai_key, example_delim="<END>", model="gpt-4", max_tokens=1000, temperature=1, completion=None):
        openai.api_key = openai_key
        cur_prompt = [{'role': "system", 'content': final_prompt}]
        the_tuple = None

        response = openai.ChatCompletion.create(messages=cur_prompt, model=model, max_tokens=max_tokens, temperature=temperature)['choices'][0]['message']['content']
        print(response)
        the_data = response.split("\n")
        json_list = []
        for data in the_data:
            idx = data.index('{')
            json_data = data[idx:]
            json_list.append(json_data)
        print(json_list) 
        # the_data = json.loads(response.replace("\n", ""))
        print(the_data)
        if completion is None:
            try:
                the_data = json.loads(response.replace("\n", ""))
                prompt = the_data["prompt"]
                completion_response = self.optimizer.call_big(prompt, max_tokens=max_tokens)
                print(completion_response)
                the_tuple = (prompt, completion_response+example_delim)
            except Exception as e:
                raise Exception(f"An error has ocurred: {e}")
                      
        else:
            try:
                the_data = json.loads(response.replace("\n", ""))
                prompt = the_data["prompt"]
                completion_response = completion.complete(prompt)
                the_tuple = (prompt, completion_response+example_delim)
            except Exception as e:
                raise Exception(f"An error has ocurred: ${e}")
            
        return the_tuple