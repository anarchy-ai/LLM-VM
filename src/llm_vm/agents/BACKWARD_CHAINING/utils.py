import requests
import traceback
import os
import random
import concurrent.futures
import openai
import urllib.request
import json
import string
import re
import sys
import threading
import time

def asyncStart(foo):
    t = (None, None)
    def new_thread():
        t[0] = foo()
    t[1] = threading.Thread(target=new_thread)
    return t

def asyncAwait(t):
    t[1].join()
    return t[0]

def flatten(a):
    return sum(a, [])

def print_op(*kargs, **kwargs):
    print(*kargs, **kwargs, flush=True)


def prepPrintPromptContext(p):
    return "--> "+p.replace("\n", "\n--> ") if len(p) > 0 else ""

def MSG(u, c):
    return [{'role': u, 'content' : c}]


context_store = {}
MIN_TRAIN_EXS = 20
MAX_TRAIN_EXS = 2000

def call_OptGPT(state, stable_context, dynamic_prompt, **kargs):
    global context_store

    prompt = stable_context + dynamic_prompt
    checkpoint = context_store[stable_context] if stable_context in context_store else ([], None, False)

    completion = None
    if checkpoint[1] is not None:
        completion = call_gpt(state, prompt, model=checkpoint[1], **kargs)

    if completion is None or len(training_exs[0]) < MAX_TRAIN_EXS:
        def promiseCompletion():
            best_completion = call_ChatGPT(state, MSG("system", prompt), gpt4=True, **kargs)
            new_data = checkpoint[0] + [(dymaic_prompt, best_completion)]
            context_store[stable_context] = (new_data, checkpoint[1], context_store[stable_context][2])

            if len(training_exs) >= MIN_TRAIN_EXS and not context_store[stable_context][2]:
                context_store[stable_context][2] = True
                def train_with():
                    old_model = context_store[stable_context][1]

                    fine_tuning_job = openai.FineTune.create(
                        training_file=os.path.abspath(training_file),
                        validation_file=os.path.abspath(validation_file))

                    job_id = fine_tuning_job["id"]
                    print(f"Fine-tuning job created with ID: {job_id}")
                    
                    while True:
                        fine_tuning_status = openai.FineTune.get_status(job_id)
                        status = fine_tuning_status["status"]
                        print(f"Fine-tuning job status: {status}")
                        if status in ["completed", "failed"]:
                            break
                        time.sleep(60)

                    new_model = fine_tuning_status["fine_tuned_model_id"]

                    context_store[stable_context][1] = new_model
                    context_store[stable_context][2] = False
                    if old_model is not None:
                        openai.Model.delete(old_model)

                asyncStart(train_with)

            return best_completion
            
        best_completion = asyncStart(promiseCompletion) # async?
        if completion is None:
            completion = asyncAwait(best_completion)
    
    return completion
    

def call_ChatGPT(state, cur_prompt, stop = None, max_tokens = 20, temperature = 0.2, gpt4 = False):
    if state.verbose > 1:
        print_op("\nGPT input for {" +str(stop) + "} "+ str(len(cur_prompt)) + ".")
    if state.verbose > 2:
        print_op(str(cur_prompt))
        
    ppt = 0.002 if not gpt4 else 0.02

    def calcCost(p):
        chars = sum((len(a['content']) for a in p))
        if state.verbose > 0:
            print_op("ASK_CHARS:", chars)
        c = (chars / 2700.0) * ppt
        if state.verbose > 2:
            print_op("PrePrice: $"+str(c))
        return c

    cost = calcCost(cur_prompt)
    try:
        ans = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301" if not gpt4 else 'gpt-4',
            max_tokens=max_tokens,
            stop=stop,
            messages=cur_prompt,
            temperature=temperature)

    except Exception as e:
        state.price += cost
        traceback.print_exc()
        print_op("Error:", e)

        return "OpenAI is down!"

    price = ppt * ans['usage']['total_tokens'] / 1000
    if state.verbose > 0:
        print_op("Price: $"+str(price))
    state.price += price
    response_text = ans['choices'][0]['message']['content']

    if state.verbose > 2:
        print_op("GPT output:")
        print_op(prepPrintPromptContext(response_text))
        print_op("GPT output fin.\n")

    return response_text



def call_gpt(state, cur_prompt: str, stop: str, max_tokens = 20, quality = "best", temperature = 0.0):
    if state.verbose > 1:
        print_op("\nGPT input for {" +stop + "} "+ str(len(cur_prompt)) + ".")
    if state.verbose > 2:
        print_op(prepPrintPromptContext(cur_prompt))

    ask_tokens = max_tokens + len(cur_prompt) / 2.7

    if state.verbose > 0:
        print_op("ASK_TOKENS:", ask_tokens)

    if (ask_tokens) > 2049:
        quality = 'best'

    model = { 'best' : ("text-davinci-003", 0.02), 
              'okay' : ("text-curie-001", 0.002), 
             }[quality]

    def calcCost(p):
        return (len(p) / 2700.0) * model[1]

    cost = calcCost(cur_prompt)

    try:
        ans = openai.Completion.create(
            model=model[0],
            max_tokens=max_tokens,
            stop=stop,
            prompt=cur_prompt,
            temperature=temperature
        )
    except Exception as e:
        print_op("WTF:", e)
        state.price += cost
        return "OpenAI is down!"

    response_text = ans['choices'][0]['text']

    simpleprice = model[1] * ans['usage']['total_tokens'] / 1000
    if state.verbose > 0:
        print_op("SimplePrice: $"+str(simpleprice))
    state.price += simpleprice

    if state.verbose > 2:
        print_op("GPT output:")
        print_op(prepPrintPromptContext(response_text))
        print_op("GPT output fin.\n")

    return response_text

def deep_fmap(lambdaFunc, json_data):
    if isinstance(json_data, list):
        return list(map(lambda listItem: deep_fmap(lambdaFunc, listItem), json_data))
    elif isinstance(json_data, tuple):
        return tuple(map(lambda tupleItem: deep_fmap(lambdaFunc, tupleItem), json_data))
    elif isinstance(json_data, dict):
        return {lambdaFunc(k): deep_fmap(lambdaFunc, v) for k, v in json_data.items()}
    else:
        return lambdaFunc(json_data)
