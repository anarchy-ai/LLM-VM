import json
import sys


class DataSynthesis:
    def __init__(self, variance, examples_to_generate):
        self.variance = variance
        self.examples_to_generate = examples_to_generate

    def data_synthesis(
        self, optimizer, prompt, response, example_delim="<Datum-Separator/>", **kwargs
    ):
        """
        This method generates QA pairs using the larger LLM to be used as training data for fine-tuning the smaller LLM.

        Parameters
        ----------
        - optimizer (class): The Optimizer class to use for fine-tuning. Could be either LocalOptimizer or HostedOptimizer.
        - prompt (str): A question to be used as a one-shot QA example for the larger LLM prompt.
        - response (str): A verified answer to the provided prompt question to be used in the one-shot QA example.
        - example_delim (str): A unique XML tag used to separate the generated JSON examples.
        - **kwargs: Additional keyword arguments to be passed into the `call_big` method.

        Returns
        ----------
        - List: A list of tuples containing the QA pairs to be used for fine-tuning.

        """
        final_prompt = (
            '{"prompt": "'
            + prompt
            + '"  , "response": "'
            + response
            + '" }'
            + "\nGenerate "
            + str(self.examples_to_generate)
            + f""" more JSONS each with a prompt and response field like the given one. 
            The content of the prompt and response fields must be similar to the given JSON. 
            Separate each JSON with the XML tag {example_delim}."""
        )
        data = None
        response = optimizer.call_big(final_prompt, **kwargs)
        datapoints = []
        print(response, file=sys.stderr)
        split_response = response.split(sep=example_delim)
        print(
            f"Generated {len(split_response)}/{self.examples_to_generate} examples.",
            file=sys.stderr,
        )
        datum_failure = 0
        bad_key_failure = 0
        resp_filter = {}
        for d in split_response:
            try:
                the_data = json.loads(d)
                the_tuple = (the_data["prompt"], the_data["response"])
                if the_tuple in resp_filter:
                    continue  # dont save a response if its already happened
                resp_filter[
                    the_tuple
                ] = True  # for now we're treating the (Q,A) pair as a single value
                datapoints.append(the_tuple)
            except json.decoder.JSONDecodeError as err:
                print(
                    f"data_synthesis response parsing failed with: { err } \n Expected a valid JSON Object but received {d}",
                    file=sys.stderr,
                )
                datum_failure += 1
            except LookupError as err:  # i have no evidence that this will happen
                print(
                    f"data_synthesis key lookup failed with: { err }", file=sys.stderr
                )
                bad_key_failure += 1
        print(
            f"Out of { len(split_response)} response objects, {datum_failure} were not valid json \n\
            and {bad_key_failure} were missing a key",
            file=sys.stderr,
        )
        return datapoints
