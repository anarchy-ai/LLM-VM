import json
import sys 
from sentence_transformers import SentenceTransformer, util


class DataSynthesis:
     def __init__(self, variance, examples_to_generate):
         self.variance = variance
         self.examples_to_generate = examples_to_generate
     def data_synthesis(self, optimizer, prompt, response, example_delim="<Datum-Separator/>", semantic_sim=True, **kwargs):
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
        final_prompt = None
        if type(prompt) is str:
            final_prompt = '{"prompt": "' +prompt+'"  , "response": "' +response+'" }'+ \
                '\nGenerate '+str(self.examples_to_generate)+F""" more JSONS each with a prompt and response field like the given one. 
                The content of the prompt and response fields must be similar to the given JSON. 
                Separate each JSON with the XML tag {example_delim}."""
        elif type(prompt) is list:
            json_str = ""
            for idx,p in enumerate(prompt):
               example_str = '{"prompt": "' + p +'"  , "response": "' + response[idx] +'" } \n'
               json_str += example_str
            final_prompt = json_str + 'Generate '+str(self.examples_to_generate)+F""" more JSONS each with a prompt and response field like the given examples. 
                The content of the prompt and response fields must be similar to the given JSON. 
                Separate each JSON with the XML tag {example_delim}."""

        data = None       
        response=optimizer.call_big(final_prompt , **kwargs)
        datapoints = []
        print(response,file=sys.stderr)
        split_response = response.split(sep=example_delim)
        print(f"Generated {len(split_response)}/{self.examples_to_generate} examples.", file=sys.stderr )
        if semantic_sim:
            num_responses = len(split_response)
            batch_responses = []
            for idx in range(0, len(split_response), 10):
                batch_str = None
                if len(split_response) - idx >= 10:
                    batch_str = "".join(split_response[idx : idx + 10])
                else:
                    batch_str = "".join(split_response[idx : len(split_response)])
                batch_responses.append(batch_str)
            embeddings = model.encode(batch_responses, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings, embeddings)
            duplicate_idx = []
            for row_idx, row in enumerate(cosine_scores):
                for i, score in enumerate(row):
                    if score >= self.variance and score < 0.99:
                        duplicate_idx.append((row_idx, i))

            deleted_idx = []
            for duplicate in duplicate_idx:
                example_idx, match_idx = duplicate
                if example_idx in deleted_idx or match_idx in deleted_idx:
                    continue
                else:
                    split_idx = (example_idx + 1) * 10
                    if split_idx < len(split_response):
                        del split_response[split_idx - 10 : split_idx]
                    else:
                        del split_response[split_idx - 10 : len(split_response)]
                    deleted_idx.append(example_idx)

            print(
                f"Found {len(split_response)} valid examples. Removed {num_responses - len(split_response)} duplicate examples.",
                file=sys.stderr,
            )
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
                print(F'data_synthesis response parsing failed with: { err } \nExpected a valid JSON Object but received {type(d)} of length {len(d)}',file=sys.stderr)
                datum_failure+=1
            except LookupError as err : # i have no evidence that this will happen
                print(F'data_synthesis key lookup failed with: { err }',file=sys.stderr)
                bad_key_failure +=1
        print(F'Out of { len(split_response)} response objects, {datum_failure} were not valid json \n\
            and {bad_key_failure} were missing a key',file=sys.stderr)
        return datapoints
