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



def flatten(a):
    return sum(a, [])


def print_op(*kargs, **kwargs):
    print(*kargs, **kwargs, flush=True, file=sys.stderr)


def prepPrintPromptContext(p):
    return "--> " + p.replace("\n", "\n--> ") if len(p) > 0 else ""


def MSG(u, c):
    return [{"role": u, "content": c}]


def call_ChatGPT(state, cur_prompt, stop=None, max_tokens=20, temperature=0.2):
    if state.verbose > 1:
        print_op("\nGPT input for {" + str(stop) + "} " + str(len(cur_prompt)) + ".")
    if state.verbose > 2:
        print_op(str(cur_prompt))

    ppt = 0.002

    def calcCost(p):
        chars = sum((len(a["content"]) for a in p))
        if state.verbose > 0:
            print_op("ASK_CHARS:", chars)
        c = (chars / 2700.0) * ppt
        if state.verbose > 2:
            print_op("PrePrice: $" + str(c))
        return c

    cost = calcCost(cur_prompt)
    try:
        ans = openai.chat.completions.create(model="gpt-3.5-turbo-0301",
        max_tokens=max_tokens,
        stop=stop,
        messages=cur_prompt,
        temperature=temperature)

    except Exception as e:
        state.price += cost
        traceback.print_exc()
        print_op("Error:", e)

        return "OpenAI is down!"

    price = ppt * ans["usage"]["total_tokens"] / 1000
    if state.verbose > 0:
        print_op("Price: $" + str(price))
    state.price += price
    response_text = ans["choices"][0]["message"]["content"]

    if state.verbose > 2:
        print_op("GPT output:")
        print_op(prepPrintPromptContext(response_text))
        print_op("GPT output fin.\n")

    return response_text


def call_gpt(
    state, cur_prompt: str, stop: str, max_tokens=20, quality="best", temperature=0.0
):
    if state.verbose > 1:
        print_op("\nGPT input for {" + stop + "} " + str(len(cur_prompt)) + ".")
    if state.verbose > 2:
        print_op(prepPrintPromptContext(cur_prompt))

    ask_tokens = max_tokens + len(cur_prompt) / 2.7

    if state.verbose > 0:
        print_op("ASK_TOKENS:", ask_tokens)

    if (ask_tokens) > 2049:
        quality = "best"

    model = {
        "best": ("text-davinci-003", 0.02),
        "okay": ("text-curie-001", 0.002),
    }[quality]

    def calcCost(p):
        return (len(p) / 2700.0) * model[1]

    cost = calcCost(cur_prompt)

    try:
        ans = openai.completions.create(model=model[0],
        max_tokens=max_tokens,
        stop=stop,
        prompt=cur_prompt,
        temperature=temperature)
    except Exception as e:
        print_op("WTF:", e)
        state.price += cost
        return "OpenAI is down!"

    response_text = ans["choices"][0]["text"]

    simpleprice = model[1] * ans["usage"]["total_tokens"] / 1000
    if state.verbose > 0:
        print_op("SimplePrice: $" + str(simpleprice))
    state.price += simpleprice

    if state.verbose > 2:
        print_op("GPT output:")
        print_op(prepPrintPromptContext(response_text))
        print_op("GPT output fin.\n")

    return response_text


def deep_fmap(lambdaFunc, json_data):
    print_op(json_data)
    if isinstance(json_data, list):
        print_op("##LIST")
        return list(map(lambda listItem: deep_fmap(lambdaFunc, listItem), json_data))
    elif isinstance(json_data, tuple):
        print_op("##TUPLE")
        return tuple(map(lambda tupleItem: deep_fmap(lambdaFunc, tupleItem), json_data))
    elif isinstance(json_data, dict):
        print_op("##DICT")
        return {lambdaFunc(k): deep_fmap(lambdaFunc, v) for k, v in json_data.items()}
    else:
        print_op("##SIMPLE")
        return lambdaFunc(json_data)


def replace_variables_for_values(
    my_dict: dict, dynamic_keys, ignore_key: str = "_______"
):
    replaced_dict = {}
    for key, value in my_dict.items():
        if key == ignore_key:
            continue
        formatted_key = key.format(**dynamic_keys)
        if isinstance(value, dict):
            formatted_value = replace_variables_for_values(value, dynamic_keys)
        elif isinstance(value, list):
            formatted_value = []
            for item in value:
                formatted_value += replace_variables_for_values(item, dynamic_keys)
        else:
            try:
                formatted_value = value.format(**dynamic_keys)
            except:
                formatted_value = value
        replaced_dict[formatted_key] = formatted_value
    return replaced_dict


def tool_api_call(self, tool, gpt_suggested_input, question, memory, facts, query=""):
    if query == "":
        query = question

    if gpt_suggested_input[0] != "{":
        gpt_suggested_input = "{" + gpt_suggested_input
    if gpt_suggested_input[-1] != "}":
        gpt_suggested_input += "}"

    if self.verbose > 1:
        print_op("GPT SUGGESTED INPUT:", gpt_suggested_input)

    parsed_gpt_suggested_input = json.loads(gpt_suggested_input)
    # make sure all of the suggested fields exist in the tool desc
    for i in parsed_gpt_suggested_input.keys():
        if i not in tool["dynamic_params"].keys():
            raise Exception("Bad Generated Input")
    tool_args = replace_variables_for_values(tool["args"], parsed_gpt_suggested_input)

    url = tool_args["url"]
    if self.verbose > -1:
        print_op(tool["method"] + ":", url)

    if "auth" in tool_args and isinstance(tool_args["auth"], dict):
        auths = list(tool_args["auth"].items())
        if len(auths) > 0:
            tool_args["auth"] = list(tool_args["auth"].items())[0]
        else:
            del tool_args["auth"]

    # Remove those parameters that are not part of Python's `requests` function.
    tool_args.pop("jsonParams", None)
    tool_args.pop("urlParams", None)

    if self.verbose > -1:
        print_op("ARGS: ", tool_args)

    resp = (requests.get if tool["method"] == "GET" else requests.post)(**tool_args)

    # print_op("FINAL URL: (" + tool["method"] + ") ", resp.url)

    actual_call = str(tool_args)

    if resp.status_code in [404, 401, 500]:
        er = " => " + str(resp.status_code)
        return "This tool isn't working currently" + er

    ret = str(resp.text)
    if self.verbose > 4:
        print(ret, file=sys.stderr)
    try:
        ret = str(json.loads(ret))
    except:
        pass

    if len(ret) > 10000:
        ret = ret[0:10000]
    mem = "".join(
        [
            self.makeInteraction(p, a, "P", "AI", INTERACTION="Human-AI")
            for p, a in memory
        ]
    ) + "".join(
        [self.makeInteraction(p, a, "P", "AI", INTERACTION="AI-AI") for p, a in facts]
    )

    prompt = MSG("system", "You are a good and helpful bot" + self.bot_str)
    prompt += MSG(
        "user",
        mem
        + "\nQ:"
        + query
        + "\n An api call about Q returned:\n"
        + ret
        + "\nUsing this information, what is the answer to Q?",
    )
    a = call_ChatGPT(self, prompt, stop="</AI>", max_tokens=256).strip()
    return a
