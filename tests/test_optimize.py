from dotenv import load_dotenv
import os
import openai
import sys

from llm_vm.completion.optimize import *

haskell = '''
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
def delete_file(file_name):
    location = os.getcwd()
    path = os.path.join(location, file_name)
    os.remove(path)
    return True
def call_ChatGPT(cur_prompt, stop = None, max_tokens = 20, temperature = 0.2, gpt4 = False):
    ans = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301" if not gpt4 else 'gpt-4',
        max_tokens=max_tokens,
        stop=stop,
        messages=cur_prompt,
        temperature=temperature)
    return ans.choices[0].message.content
    return response_text

def call_gpt(cur_prompt: str, stop: str, max_tokens = 20, quality = "best", temperature = 0.0, model = "text-davinci-003"):
    ans = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        prompt=cur_prompt,
        temperature=temperature
    )
    return ans['choices'][0]['text']
import gzip
import json
def create_jsonl_file(data_list: list, file_name: str, compress: bool = True) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not file_name.endswith(sjsonl):
        file_name = file_name + sjsonl
    # Save data
    if compress:
        file_name = file_name + sgz
        with gzip.open(file_name, 'w') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(file_name, 'w') as out:
            for ddict in data_list:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)
    return file_name, open(file_name, "rb")
'''

def run_test_stub():
    try:
        load_dotenv()
    except:
        pass
    openai_api_key = os.getenv('LLM_VM_OPENAI_API_KEY')
    openai.api_key =openai_api_key
    # anarchy_key = os.getenv('LLM_VM_ANARCHY_KEY')
    print("key:", openai.api_key[0:5], file=sys.stderr)
    optimizer = LocalOptimizer(MIN_TRAIN_EXS=1,openai_key=openai_api_key)
    #optimizer = HostedOptimizer(openai_key = openai.api_key,
    #                            anarchy_key = anarchy_key,
    #                            MIN_TRAIN_EXS=2)
    i = 0
    optimizer.complete("Answer question Q. ","Q: What is the currency in myanmmar", \
                 temperature = 0.0, data_synthesis = True,\
                 min_examples_for_synthesis=0,finetune=True)

if __name__ == "__main__":
    run_test_stub()
    '''
    for h in haskell.splitlines():
        print("At: ", i)
        try:
            print(optimizer.complete("Please convert this line to some haskell:", h + "\nHaskell:", max_tokens = 100, temperature = 0.0))
        except Exception as e:
            print('E:', e)

        time.sleep(2)
        if i > 3 and i < 20:
            time.sleep(120)
        i += 1
    '''
