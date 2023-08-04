import json
import sys
from sentence_transformers import SentenceTransformer, util
import openai


class DataSynthesis:
     def __init__(self, variance, examples_to_generate):
         self.variance = variance
         self.examples_to_generate = examples_to_generate
     def data_synthesis(self, optimizer, prompt, response, example_delim="<Datum-Separator/>", openai_key=None, semantic_sim=True, **kwargs):
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
        model = SentenceTransformer("all-MiniLM-L6-v2")
        datapoints = []
        final_prompt = None
        if type(prompt) is str:
            final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'
        final_prompt = "Generate 1 json similar to the one below. \n" + final_prompt
        while len(datapoints) < self.examples_to_generate:
            openai.api_key=openai_key
            cur_prompt = [{'role': "system", 'content' : final_prompt}]
            response=openai.ChatCompletion.create(messages=cur_prompt,model="gpt-4",max_tokens=1000,temperature=1)['choices'][0]['message']['content']
        
            try:
                the_data = json.loads(response.replace("\n",""))
                the_tuple = (the_data["prompt"],the_data["response"])
                datapoints.append(the_tuple)
            except:
                pass
    
        return datapoints
