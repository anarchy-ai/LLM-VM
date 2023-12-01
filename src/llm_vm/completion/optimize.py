import openai
import traceback
import threading
import time
import os
import sys
import signal
import json
import tempfile
import abc
import requests
import hashlib
import pickle
from llm_vm.guided_completion import Completion
# we need to package-ify so this works
from llm_vm.completion.data_synthesis import DataSynthesis
import inspect
from __future__ import annotations


from typing import Optional


from google.auth import default
from google.cloud import aiplatform
import pandas as pd
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import TuningEvaluationSpec


credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])



def tuning(
    project_id: str,
    location: str,
    model_display_name: str,
    training_data: pd.DataFrame | str,
    train_steps: int = 10,
    evaluation_dataset: Optional[str] = None,
    tensorboard_instance_name: Optional[str] = None,
) -> TextGenerationModel:
    """Tune a new model, based on a prompt-response data.
    "training_data" can be either the GCS URI of a file formatted in JSONL format
    (for example: training_data=f'gs://{bucket}/{filename}.jsonl'), or a pandas
    DataFrame. Each training example should be JSONL record with two keys, for
    example:
      {
        "input_text": <input prompt>,
        "output_text": <associated output>
      },
    or the pandas DataFame should contain two columns:
      ['input_text', 'output_text']
    with rows for each training example.
    Args:
      project_id: GCP Project ID, used to initialize vertexai
      location: GCP Region, used to initialize vertexai
      model_display_name: Customized Tuned LLM model name.
      training_data: GCS URI of jsonl file or pandas dataframe of training data.
      train_steps: Number of training steps to use when tuning the model.
      evaluation_dataset: GCS URI of jsonl file of evaluation data.
      tensorboard_instance_name: The full name of the existing Vertex AI TensorBoard instance:
        projects/PROJECT_ID/locations/LOCATION_ID/tensorboards/TENSORBOARD_INSTANCE_ID
        Note that this instance must be in the same region as your tuning job.
    """
    vertexai.init(project=project_id, location=location, credentials=credentials)
    eval_spec = TuningEvaluationSpec(evaluation_data=evaluation_dataset)
    eval_spec.tensorboard = aiplatform.Tensorboard(
        tensorboard_name=tensorboard_instance_name
    )
    model = TextGenerationModel.from_pretrained("text-bison@001")

    model.tune_model(
        training_data=training_data,
        # Optional:
        model_display_name=model_display_name,
        train_steps=train_steps,
        tuning_job_location="europe-west4",
        tuned_model_location=location,
        tuning_evaluation_spec=eval_spec,
    )

    print(model._job.status)

    return model


if __name__ == "__main__":
    tuning()


job_id = None # we want to be able to cancel a fine_tune if you kill the program

def exit_handler(signum, frame):

    if (None != job_id):
        print("cancelling fine-tune if applicable", file=sys.stderr)
        openai.fine_tunes.cancel(id=job_id)

    print("user interrupt, exiting", file=sys.stderr)
    sys.exit()


if threading.current_thread() == threading.main_thread():        
    signal.signal(signal.SIGINT, exit_handler)


def generate_hash(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(input_string).encode('utf-8'))
    return int(sha256_hash.hexdigest(), 16) % 10**18

def asyncStart(foo):
    t = [None, None]
    def new_thread():
        if callable(foo):
            t[0] = foo()
        else:
            t[0] = lambda x: x
    t[1] = threading.Thread(target=new_thread)
    t[1].start()
    return t

def asyncAwait(t):
    t[1].join()
    return t[0]


class LocalEphemeral:

    def __init__(self):
        self.training_store = {}

    def get_data(self, c_id):
        self.init_if_null(c_id)
        return self.training_store[c_id]["data"]

    def load_data(self,file):
        #using pickle files right now, in the future we will have to use databases
        self.training_store = pickle.load(file)

    def store_data(self,file):
         #using pickle files right now, in the future we will have to use databases
        pickle.dump(self.training_store,file)

    def add_example(self, c_id, example):
        self.init_if_null(c_id)
        self.training_store[c_id]["data"] += [example]

    def set_training_in_progress(self, c_id, is_training):
        self.init_if_null(c_id)
        self.training_store[c_id]["is_training"] = is_training

    def get_training_in_progress_set_true(self, c_id):
        self.init_if_null(c_id)
        # TODO: this could cause a concurrency bug when distributed!
        self.training_store[c_id]['lock'].acquire()
        old_val = self.training_store[c_id]["is_training"]
        if not old_val:
            self.training_store[c_id]["is_training"] = True
        self.training_store[c_id]['lock'].release()
        return old_val

    def set_model(self, c_id, model_id):
        self.init_if_null(c_id)
        self.training_store[c_id]["model"] = model_id

    def get_model(self, c_id):
        self.init_if_null(c_id)
        return self.training_store[c_id]["model"]

    def init_if_null(self, c_id):
        if not c_id in self.training_store:
            self.training_store[c_id] = { "is_training": False,
                                          'lock' : threading.Lock(),
                                          "data": [],
                                          "model": None }



class Optimizer:
    @abc.abstractmethod
    def complete(self, stable_context, dynamic_prompt, **kwargs):
        """
        Runs a completion using the string stable_context+dynamic_prompt.  Returns an optional training closure to use if the
        caller decides that the completion was particularly good.

        This method first checks if a model exists for the stable_context. If it does, it uses the model to complete the prompt.
        It then checks if the number of training examples is less than the maximum allowable. If it is, or if a model wasn't
        previously found, it retrieves the best completion for the prompt using a larger model, adds a new datapoint for training,
        and potentially fine-tunes a new model using the updated data, storing the new model if successful.

        The function returns the best completion (either generated by the stored model or the larger model).

        This can not handle cases where either stable_context or dynamic_prompt are just whitespace!

        Parameters:
        ----------
        - stable_context (str): Stable contextual data to use as a basis for training.
        - dynamic_prompt (str): The dynamic data to generate a completion for and potentially add to training data.
        - **kwargs: Additional arguments to be passed to the `call_small` and `call_big` methods.

        Returns:
        ----------
        - completion (str): The best completion for the dynamic prompt, as generated by either the stored model or the larger model.
        """


class HostedOptimizer(Optimizer):
    def __init__(self, anarchy_key, openai_key, MIN_TRAIN_EXS=20, MAX_TRAIN_EXS = 2000):
        self.anarchy_key = anarchy_key
        self.openai_key = openai_key
        self.MIN_TRAIN_EXS = MIN_TRAIN_EXS
        self.MAX_TRAIN_EXS = MAX_TRAIN_EXS

    def complete(self, stable_context, dynamic_prompt, **kwargs):
        """
        TODO: Runs the optimizing completion process on anarchy's hosted server with persistence.

        Parameters:
        ----------
        - stable_context (str): Stable contextual data to use as a basis for training.
        - dynamic_prompt (str): The dynamic data to generate a completion for and potentially add to training data.
        - **kwargs: Additional arguments to be passed to the `call_small` and `call_big` methods.

        Returns:
        ----------
        - completion (str): The best completion for the dynamic prompt, as generated by either the stored model or the larger model.
        """
        url = "https://api.chat.dev/completion/optimizing"
        payload = {**kwargs,
                   'stable_context': stable_context,
                   'dynamic_prompt': dynamic_prompt,
                   'anarchy_key' : self.anarchy_key,
                   'openai_key' : self.openai_key,
                   'MIN_TRAIN_EXS' : self.MIN_TRAIN_EXS,
                   'MAX_TRAIN_EXS' : self.MAX_TRAIN_EXS
                   }
        headers = {'Authorization': f'Bearer {self.anarchy_key}'}

        print("Payload: ", payload, file=sys.stderr)
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for 4XX and 5XX status codes
            return response.json()['completion']
        except requests.exceptions.RequestException as e:
            print("Error occurred:", e, file=sys.stderr)


class LocalOptimizer(Optimizer):
    def __init__(self, storage=LocalEphemeral(), MIN_TRAIN_EXS = 1, MAX_TRAIN_EXS = 2000, call_small = None , call_big = None , big_model = None, small_model = None, openai_key="", variance=0.87, num_examples=50):
        self.storage = storage
        self.MIN_TRAIN_EXS = MIN_TRAIN_EXS
        self.MAX_TRAIN_EXS = MAX_TRAIN_EXS
        self.call_small = call_small
        self.call_big = call_big
        self.big_model = big_model
        self.small_model = small_model
        self.openai_key = openai_key
        self.data_synthesizer = DataSynthesis(variance, num_examples)

    def complete(self, stable_context, dynamic_prompt, data_synthesis = False, finetune = False, regex = None, type = None, choices = None, **kwargs):

        openai.api_key = self.openai_key
        completion, train = self.complete_delay_train(stable_context, dynamic_prompt, run_data_synthesis=data_synthesis, regex = regex, choices = choices, type = type, **kwargs)
        if finetune:
            train()
        return completion

    def complete_delay_train(self, stable_context, dynamic_prompt, run_data_synthesis = False, min_examples_for_synthesis = 1 ,c_id = None, regex = None, choices = None, type = None, grammar_type = None, **kwargs):
        """
        Runs a completion using the string stable_context+dynamic_prompt.  Returns an optional training closure to use if the
        caller decides that the completion was particularly good.

        This method first checks if a model exists for the stable_context. If it does, it uses the model to complete the prompt.
        It then checks if the number of training examples is less than the maximum allowable. If it is, or if a model wasn't
        previously found, it retrieves the best completion for the prompt using a larger model, adds a new datapoint for training,
        and potentially fine-tunes a new model using the updated data, storing the new model if successful.

        The function returns the best completion (either generated by the stored model or the larger model), and a closure
        function that encapsulates the fine-tuning process for potential execution at a later time.

        Parameters:
        ----------
        - stable_context (str): Stable contextual data to use as a basis for training.
        - dynamic_prompt (str): The dynamic data to generate a completion for and potentially add to training data.
        - c_id (str): To be used if multiple users could share the same stable_contexts so that we don't leak data.  If its None, defaults to all possible context.
        - **kwargs: Additional arguments to be passed to the `call_small` and `call_big` methods.

        Returns:
        ----------
        - completion (str): The best completion for the dynamic prompt, as generated by either the stored model or the larger model.
        - succeed_train (function): A closure function that encapsulates the fine-tuning process, ready for
        execution at a later time.  If you pass it a completion, it will use that, otherwise it will use the completion from the "best" model.
        """
        assert dynamic_prompt.strip() != "" or stable_context.strip() != ""
        assert self.call_big is not None and self.call_small is not None and self.big_model is not None and self.small_model is not None
        if stable_context.strip() == "" :
            print("Running with an empty context", file=sys.stderr)

        prompt = (stable_context + dynamic_prompt).strip()
        c_id_repr = str({'stable_context' : stable_context,
                    'args' : kwargs,
                    'MIN_TRAIN_EXS' : self.MIN_TRAIN_EXS,
                    'MAX_TRAIN_EXS' : self.MAX_TRAIN_EXS,
                    'call_small' : str(self.small_model).split(' ')[0], # HACKS
                    'call_big' : str(self.big_model).split(' ')[0],
                    }) if c_id is None else c_id
        c_id = generate_hash(c_id_repr)
        completion = None
        # will return None if no constraint is specified
        completion_model = Completion.create(regex, type, choices, grammar_type)

        model = self.storage.get_model(c_id)
        # this gives us the model_id
        if model is not None and completion_model is None:
            print("Using the new model:", model, flush=True, file=sys.stderr)
            completion = self.call_small(prompt = dynamic_prompt.strip(), model=model, **kwargs)

        training_exs = self.storage.get_data(c_id)

        best_completion_promise = None
        succeed_train = None
        best_completion = None

        if len(training_exs) < self.MAX_TRAIN_EXS:
            def promiseCompletion():
                if completion_model is not None:
                    best_completion = completion_model.complete(prompt)
                else:
                    best_completion = self.call_big(prompt, **kwargs)

                def actual_train(use_completion = None):

                    train_completion = best_completion if use_completion is None else use_completion
                    new_datapoint = (dynamic_prompt.strip(), train_completion)
                    self.storage.add_example(c_id, new_datapoint)
                    small_model_filename = kwargs.get("small_model_filename", None)

                    if run_data_synthesis:
                        if len(self.storage.get_data(c_id)) < min_examples_for_synthesis:
                            print("Data synthesis is not available right now, need more examples in storage.", file=sys.stderr)
                        else:
                            for j in self.data_synthesizer.data_synthesis(self.call_big, prompt,best_completion,openai_key=self.openai_key, completion=completion_model, **kwargs):
                                self.storage.add_example(c_id, j)
                    training_exs = self.storage.get_data(c_id)
                    print(training_exs, file=sys.stderr)
                    print("Considering Fine-tuning", flush=True, file=sys.stderr)

                    if len(training_exs) >= self.MIN_TRAIN_EXS and not self.storage.get_training_in_progress_set_true(c_id):
                        print("Actually Fine-tuning", flush=True, file=sys.stderr)
                        print("Training examples:",str(len(training_exs)), file=sys.stderr)
                        asyncStart(self.small_model.finetune(training_exs,self,c_id,small_model_filename))
                return (best_completion, actual_train)

            best_completion_promise = asyncStart(promiseCompletion)

            if completion is None:
                # crazy story: succeed_train gets set before this anyway if it makes sense to set it!
                completion, succeed_train = asyncAwait(best_completion_promise)

            else:
                _, succeed_train = asyncAwait(best_completion_promise)



        def succeed_train_closure(use_completion = None):
            def promise():
                if succeed_train is not None:
                    return succeed_train(use_completion)
                if best_completion_promise is not None:
                    try:
                        return asyncAwait(best_completion)[1](use_completion)
                    except:
                        return
            return asyncStart(promise)

        return completion, succeed_train_closure




def create_jsonl_file(data_list):
    out = tempfile.TemporaryFile('w+')
    for a,b in data_list:
        out.write(json.dumps({'prompt': a, 'completion': b}) + "\n")
    out.seek(0)
    return out
