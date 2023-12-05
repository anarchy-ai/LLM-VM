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
import re

class DataSynthesis:
    def __init__(self, variance, examples_to_generate, seed_examples=10):
        self.variance = variance
        self.seed_examples = seed_examples
        self.examples_to_generate = examples_to_generate
        self.call_big = None

    def data_synthesis(self, call_big, prompt, response, openai_key=None, completion=None, seed_batch_size=10, **kwargs):
        """
        This method generates QA pairs using the larger LLM to be used as training data for fine-tuning the smaller LLM.

        Parameters
        ----------
        - optimizer (class): The Optimizer class to use for fine-tuning. Could be either LocalOptimizer or HostedOptimizer.
        - prompt (str | list ): A question to be used as a one-shot QA example for the larger LLM prompt.
        - response (str | list): A verified answer to the provided prompt question to be used in the one-shot QA example.
        - example_delim (str): A unique XML tag used to separate the generated JSON examples. Default value is "<END>".
        - semantic_sim (bool): Option to use semantic similarity to filter duplicate generations. Default value is True.
        - **kwargs: Additional keyword arguments to be passed into the `call_big` method.

        Returns
        ----------
        - List: A list of tuples containing the QA pairs to be used for fine-tuning.
        """
        self.call_big = call_big
        if os.path.isfile(conf.settings.data_gen_file):
            new_file = open(conf.settings.data_gen_file,"rb")
            return list(pickle.load(new_file))
        datapoints_list = []
        final_prompt = '{"prompt": "' + prompt +'"  , "response": "' +response+'" }'
        final_prompt = "Generate {seed_batch_size} json similar to the one below and separate each json using a new line. \n" + final_prompt
        while len(datapoints_list) < self.examples_to_generate:
            remaining_examples = self.examples_to_generate - len(datapoints_list)
            if remaining_examples < seed_batch_size:
                final_prompt = "Generate {remaining_examples} json similar to the one below and separate each json using a new line. \n" + final_prompt                
            datapoints = self.generate_examples(final_prompt, openai_key, completion=completion, call_big_kwargs=kwargs)
            datapoints_list += datapoints
            # Open AI rate limit is 1 request/second 
            time.sleep(1)

        new_file = open(conf.settings.data_gen_file,"wb")
        pickle.dump(datapoints_list, new_file)
        return datapoints_list
    
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def generate_examples(self, final_prompt, openai_key, example_delim="<END>", model="gpt-4", max_tokens=1000, temperature=1, completion=None, call_big_kwargs={}):
        # TODO: Remove duplicates from synthesized data
        openai.api_key = openai_key
        cur_prompt = [{'role': "system", 'content': final_prompt}]
        tuple_list = []

        # Generate seed prompts and responses using openAI
        response = openai.chat.completions.create(messages=cur_prompt, model=model, max_tokens=max_tokens, temperature=temperature).choices[0].message.content
        the_data = response.split("\n")
        # GPT responses might have empty strings
        the_clean_data = []
        if '' in  the_data:
            # loop to remove empty strings
            for i in range(len(the_data)):
                if the_data[i] != '':
                    the_clean_data.append(the_data[i])

        json_list = []
        for data in the_clean_data:
            # Sometimes GPT-4 returns a numbered list
            if data[0] != '{':
                idx = data.index('{')
                json_data = data[idx:]
                json_list.append(json_data)
            else:
                json_list.append(data)
        
        prompt_list = [] 

        # to use open AI prompts + responses seed_examples must be equal to examples_to_generate
        if self.examples_to_generate == self.seed_examples:
            for j in json_list:
                j_dict = json.loads(j)
                clean_res = re.sub('\n', '', j_dict["response"])
                j_tuple = (j_dict["prompt"], clean_res+example_delim)
                tuple_list.append(j_tuple)
            return tuple_list
        else:
            # use seed to generate more examples
            seed_prompts = []
            ex_per_seed = self.examples_to_generate // self.seed_examples
            for j in json_list:
                j_dict = json.loads(j)
                seed_prompts.append(j_dict["prompt"])
            for s in seed_prompts:
                s_examples = []
                while len(s_examples) < ex_per_seed:
                    gen_prompt = f"Generate 1 statement like the one below: \n" + s
                    res = self.call_big(gen_prompt, max_tokens=max_tokens, **call_big_kwargs)
                    s_examples.append(res)
                prompt_list += s_examples

        if completion is None:
            try:
                if len(prompt_list) == 0:
                    for pair in json_list:
                        example = json.loads(pair)
                        prompt = example["prompt"]
                        prompt_list.append(prompt)

                # Use example prompts with big model to generate responses with configured variance                
                for p in prompt_list:
                    completion_response = self.call_big(p, max_tokens=max_tokens, **call_big_kwargs)
                    print(completion_response)
                    the_tuple = (p, completion_response+example_delim)
                    tuple_list.append(the_tuple)
            except Exception as e:
                raise Exception(f"An error has ocurred: {e}")                     
        else:
            try:
                for pair in json_list:
                    example = json.loads(pair)
                    prompt = example["prompt"]
                    prompt_list.append(prompt)
                for p in prompt_list:
                    completion_response = completion.complete(prompt)
                    the_tuple = (p, completion_response+example_delim)
                    tuple_list.append(the_tuple)
            except Exception as e:
                raise Exception(f"An error has ocurred: ${e}")
        # use big model prompts + responses to finetune small model     
        return tuple_list