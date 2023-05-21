import openai
import traceback
import threading
import time
import os
import json
import tempfile
import abc

def asyncStart(foo):
    t = [None, None]
    def new_thread():
        t[0] = foo()
    t[1] = threading.Thread(target=new_thread)
    t[1].start()
    return t

def asyncAwait(t):
    t[1].join()
    return t[0]


class local_ephemeral:

    def __init__(self):
        self.training_store = {}

    def get_data(self, c_id):
        self.init_if_null(c_id)
        return self.training_store[c_id]["data"]

    def add_example(self, c_id, example):
        self.init_if_null(c_id)
        self.training_store[c_id]["data"] += [example]

    def set_training_in_progress(self, c_id, is_training):
        self.init_if_null(c_id)
        self.training_store[c_id]["is_training"] = is_training

    def get_training_in_progress(self, c_id):
        self.init_if_null(c_id)
        return self.training_store[c_id]["is_training"]

    def set_model(self, c_id, model_id):
        self.init_if_null(c_id)
        self.training_store[c_id]["model"] = model_id

    def get_model(self, c_id):
        self.init_if_null(c_id)
        return self.training_store[c_id]["model"]

    def init_if_null(self, c_id):
        if not c_id in self.training_store:
            self.training_store[c_id] = { "is_training": False,
                                          "data": [],
                                          "model": None }


def CALL_BIG(prompt, gpt4 = False, **kwargs):
    cur_prompt = [{'role': "system", 'content' : prompt}]
    ans = openai.ChatCompletion.create(
        messages=cur_prompt,
        model="gpt-3.5-turbo-0301" if not gpt4 else 'gpt-4',
        **kwargs)

    return ans['choices'][0]['message']['content']

def CALL_SMALL(*args, **kwargs):
    ans = openai.Completion.create(*args, **kwargs)
    return ans['choices'][0]['text']



class Optimizer:
    @abc.abstractmethod
    def complete(self, stable_context, dynamic_prompt, **kwargs):
        pass


class HostedOptimizer(Optimizer):
    def __init__(self, anarchy_key, openai_key, MIN_TRAIN_EXS = 20, MAX_TRAIN_EXS = 2000, call_small = "claude", call_big = "gpt-4"):
        self.anarchy_key = anarchy_key
        self.openai_key = openai_key
        self.MIN_TRAIN_EXS = MIN_TRAIN_EXS
        self.MAX_TRAIN_EXS = MAX_TRAIN_EXS
        self.call_small = call_small
        self.call_big = call_big

    def complete(self, stable_context, dynamic_prompt, is_calling_big, **kwargs):
        """
        Completes text optimization using either the small or big model based on the value of "is_calling_big".

        Params:
            stable_context (str): The stable context for the optimization.
            dynamic_prompt (str): The dynamic prompt for the optimization.
            is_calling_big (bool): Determines whether to use the big model or small model.
            **kwargs: Additional arguments (if they exist) to the API call.

        Returns:
            str: The completed, ouputted model text.
        """
        # Make call to OpenAI API for the text optimization and completion
        return CALL_SMALL(
            prompt=stable_context + dynamic_prompt,
            model=self.call_small,
            **kwargs
        ) if is_calling_big else CALL_BIG(
            prompt=stable_context + dynamic_prompt,
            model=self.call_big,
            **kwargs
        )
    
class LocalOptimizer(Optimizer):
    def __init__(self, storage=local_ephemeral(), MIN_TRAIN_EXS = 20, MAX_TRAIN_EXS = 2000, call_small = CALL_SMALL, call_big = CALL_BIG):
        self.storage = storage
        self.MIN_TRAIN_EXS = MIN_TRAIN_EXS
        self.MAX_TRAIN_EXS = MAX_TRAIN_EXS
        self.call_small = call_small
        self.call_big = call_big

    def complete(self, stable_context, dynamic_prompt, **kwargs):
        completion, train = self.complete_delay_train(stable_context, dynamic_prompt, **kwargs)
        train()
        return completion

    def complete_delay_train(self, stable_context, dynamic_prompt, **kwargs):
        prompt = stable_context + dynamic_prompt
        c_id = stable_context

        completion = None
        if self.storage.get_model(c_id) is not None:
            print("Using the new model!")
            completion = self.call_small(prompt, model=self.storage.get_model(c_id), **kwargs)
            
        training_exs = self.storage.get_data(c_id)
        
        succeed_train = { 'closure' : lambda: None }

        if completion is None or len(training_exs) < self.MAX_TRAIN_EXS:
            def promiseCompletion():
                best_completion = self.call_big(prompt, **kwargs)
                def actual_train():
                    new_datapoint = (dynamic_prompt, best_completion)
                    self.storage.add_example(c_id, new_datapoint)
                    training_exs = self.storage.get_data(c_id)
                    print(f"Considering Fine-tuning")
                    if len(training_exs) >= self.MIN_TRAIN_EXS and not self.storage.get_training_in_progress(c_id):
                        print(f"Actually Fine-tuning")
                        self.storage.set_training_in_progress(c_id, True)

                        def train_with():
                            old_model = self.storage.get_model(c_id)
                            training_file = create_jsonl_file(training_exs)
                            upload_response = openai.File.create(file=training_file, purpose="fine-tune")
                            training_file.close()
                            fine_tuning_job = openai.FineTune.create(training_file= upload_response.id)

                            job_id = fine_tuning_job["id"]
                            print(f"Fine-tuning job created with ID: {job_id}")

                            while True:
                                fine_tuning_status = openai.FineTune.retrieve(id=job_id)
                                status = fine_tuning_status["status"]
                                print(f"Fine-tuning job status: {status}")
                                if status in ["succeeded", "completed", "failed"]:
                                    break
                                time.sleep(60)

                            new_model_id = fine_tuning_status.fine_tuned_model

                            self.storage.set_model(c_id, new_model_id)
                            self.storage.set_training_in_progress(c_id, False)
                            if old_model is not None:
                                openai.Model.delete(old_model)

                        asyncStart(train_with)
                succeed_train['closure'] = actual_train
                return best_completion

            best_completion = asyncStart(promiseCompletion)
            if completion is None:
                # crazy story: succeed_train gets set before this anyway if it makes sense to set it!
                completion = asyncAwait(best_completion)

        return completion, succeed_train['closure']

def create_jsonl_file(data_list):
    out = tempfile.TemporaryFile('w+')
    for a,b in data_list:
        out.write(json.dumps({'prompt': a, 'completion': b}) + "\n")
    out.seek(0)
    return out
