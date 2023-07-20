import os
import sys

import requests
from llama_index import Document, GPTTreeIndex
from llm_vm.agents.FLAT.agent_helper.utils import print_op, make_interaction, print_big
from llm_vm.agents.FLAT.agent_helper.requests.call_llm import call_llm
from llm_vm.agents.FLAT.agent_helper.replacer import replace_variables_for_values
from llm_vm.utils.typings_llm import *
from llm_vm.agents.FLAT.agent_helper.bothandler import prompt_for_answer, prompt_for_instructions

# # Get the current file's directory to grab the python files with common functionality in the utils/ folder
# current_dir = os.path.dirname(os.path.abspath(__file__))
# grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# utils_dir = os.path.join(grandparent_dir, 'utils/')
# sys.path.append(utils_dir)

from llm_vm.utils.labels import *
from llm_vm.agents.FLAT.agent_helper.tools import CUSTOM_TOOL_ANSWER_EMBEDDING
import json

def __format_tool_url (url: str) -> str:
    if CUSTOM_TOOL_ANSWER_EMBEDDING in url:
        return "CHAT.DEV Embeddings Service"
    return url


def use_tool(
    tool: SingleTool,
    gpt_suggested_input: dict,
    question: str,
    memory: TupleList,
    verbose: int,
    bot_instructions: str,
    response_instructions: str = ""
) -> Tuple[str, str, Any, float]:
    price = 0

    # make sure all of the suggested fields exist in the tool desc
    for i in gpt_suggested_input.keys():
        if i not in tool["dynamic_params"].keys():
            raise Exception("Bad Generated Input")

    try:
        tool_args = replace_variables_for_values(
            tool["args"],
            gpt_suggested_input
        )
    except Exception as e:
        print_big(e, "EXCEPTION");
        tool_args = {}

    url = tool_args['url']

    if 'auth' in tool_args and isinstance(tool_args['auth'], dict):
        auths = list(tool_args['auth'].items())
        if len(auths) > 0:
            tool_args['auth'] = list(tool_args['auth'].items())[0]
        else:
            del tool_args['auth']

    # Remove those parameters that are not part of Python's `requests` function.
    tool_args.pop("jsonParams", None)
    tool_args.pop("urlParams", None)

    # print_big(tool_args, "ARGS")
    request_function = None
    method = tool['method'].upper()
    if method == "PUT":
        request_function = requests.put
    elif method == "POST":
        request_function = requests.post
    elif method == "PATCH":
        request_function = requests.patch
    elif method == "DELETE":
        request_function = requests.delete
    elif method == "GET":
        request_function = requests.get
    else:
        request_function = requests.get

    resp = request_function(**tool_args)
    print_big(resp.url, "FINAL URL: (" + tool["method"] + ") ")

    api_call_args = str(tool_args)
    return_value = str(resp.text)

    formatted_url = __format_tool_url(str(url))

    if verbose > -1:
        pass
        #commented out, causes test cases to fail because the return json of some wolfram alpha searches contains character encodings that charmap does not know.
        # leads to error being thrown.
        # print_op("URL Response:", ret)

    try:
        # seems pointless, but it formats the resulting json without spaces.
        # If the response isn't json, it does nothing.
        return_value_json = json.loads(return_value)
        return_value = str(return_value_json)
    except:
        return_value_json = {"response": return_value}
        pass

    if resp.status_code in [400, 401, 404, 500]:
        er = " → " + str(resp.status_code)
        return "This tool isn't working currently" + er, formatted_url + er, return_value_json, price


    summarised_ret: str = ""
    if len(return_value) > 8000:
        document = Document(return_value)
        try:
            if verbose > 2:
                print_op("### Summarising with GPT Tree Index ...")
            summarised_ret = GPTTreeIndex([document]).query(
                response_instructions, response_mode="tree_summarize")
        except:
            return "OpenAI is down!", formatted_url, None, price

        if verbose > 2:
            print_op("SUMMARISED:", summarised_ret)

        if str(summarised_ret).find("ANSWER: ") == 0:
            summarised_ret = str(return_value)[len("ANSWER: X."):].strip(" .")

    else:

        mem = "".join([make_interaction(p, a, INTERACTION = L_INT_HUMAN) for p,a in memory]) \

        # THIS WORKS VERY WELL, but using <XML> tags here does not work.
        summarise_prompt, stop = f"{bot_instructions}<{L_CONVERSATION}>{mem}</{L_CONVERSATION}>\n\
Description: a JSON object contains:\n{return_value}\nUsing this information, what is the [{L_ANSWER}] to [{L_QUESTION}]?\n \
{prompt_for_instructions(response_instructions)}{prompt_for_answer(question)}", None

        if verbose > 3:
            print_big(summarise_prompt, "SUMMARY OF JSON DECODE")

        summarised_ret, price = call_llm({
            "llm": LLMCallType.OPENAI_COMPLETION,
            "model": OpenAIModel.DAVINCI_TEXT.value,
            "max_tokens": 256,
            "stop": stop,
            "prompt": summarise_prompt,
            "temperature": 0.1,
        })
        print_big(summarised_ret, "SUMMARISED ANSWER")

    return summarised_ret, formatted_url, return_value_json, price
