try:    
    from .utils import *
except:
     from utils import *
     
from labels import *
import string
import re
import sys
import random

random_fixed_seed = random.Random(3)

def MSGM(u, c):
    #return [{'role': u, 'content' : c}]
    if u == "system":
        return f"<TASK>{c}</TASK>"
    if u == "user":
        return f"<GOAL>{c}</GOAL>"
    if u == "assistant":
        return f"<TOOL_ID>{c}</TOOL_ID>"

def choose_tool(state, memory, question, tools_left): 
    tools = list(enumerate(state.tools))

    def makeQuestion(question, tools_to_use):
        return f"<Q>{question}</Q>" \
            + " Which tool in ["+", ".join([str(i) for i in tools_to_use]) + "] is most applicable?" 
    
    examples = []
    for tool_id, tool in tools:
        for tool_example in tool['examples']:
            examples += ["<PICK>" + MSGM('user', makeQuestion(tool_example, range(len(tools)))) + MSGM("assistant", str(tool_id)) + "</PICK>"]

    random_fixed_seed.shuffle(examples)
    examples = "".join(examples)

    def makeToolDesc(tool_id, tool):
        return "<TOOL>" \
            + f"<{TOOL_ID}>{str(tool_id)}</{TOOL_ID}>" \
            + f"<{DESCRIPTION}>{tool['description']}</{DESCRIPTION}>" \
            + "</TOOL>"

    tool_context = "".join([makeToolDesc(t, tool) for t,tool in tools])

    mem = "".join( f"<u>{u}</u><ai>{ai}</ai>" for u,ai in memory)


    prompt = MSGM('system', tool_context + "<Objective>Decide which available tool is best to answer Q.  Always provides an answer. The lowest # tool that could be used is the tool that should be picked.</Objective>") \
           + examples \
           + "<PICK>"+MSGM('user', f"<PRIOR_CONV_HISTORY>{mem}</PRIOR_CONV_HISTORY>" + makeQuestion(question, tools_left)) + "<TOOL_ID>"

    output = call_ChatGPT(state, MSG("system",prompt), stop="</TOOL_ID>", max_tokens = 40, temperature = 0, gpt4=True).strip()

    if state.verbose > 0:
        print_op("OUTPUT:", output)
    return int(output)


'''
def choose_tool(state, question, tools_left): 
    tools = list(enumerate(state.tools))

    def makeQuestion(question, tools_to_use):
        return f"<Q>{question}</Q>" \
            + "\nWhich tool in ["+", ".join([str(i) for i in tools_to_use]) + "] is best to try first?" 
            #+ "<TOOLS>" + ", ".join([str(i) for i in tools_to_use]) + "</TOOLS> Which tool # is best to try first?" 

    examples = []
    for tool_id, tool in tools:
        for tool_example in tool['examples']:
            examples += MSG('user', makeQuestion(tool_example, range(len(tools)))) + MSG("assistant", str(tool_id))

    random_fixed_seed.shuffle(examples)

    def makeToolDesc(tool_id, tool):
        return "<TOOL>" \
            + f"<{TOOL_ID}>{str(tool_id)}</{TOOL_ID}>" \
            + f"<{DESCRIPTION}>{tool['description']}</{DESCRIPTION}>" \
            + "</TOOL>"


    tool_context = "".join([state.makeToolDesc(t) for t,_ in tools])
    prompt = MSG('system', tool_context + "<Objective>Decide which available tool is best to answer Q.  Always provides an answer. Uses tool 0 if the other tools don't fit well.</Objective>") \
           + examples \
           + MSG('user', makeQuestion(question, tools_left))

    output = call_ChatGPT(state, prompt, None, max_tokens = 40, temperature = 0, gpt4=True).strip()

    if state.verbose > 0:
        print_op("OUTPUT:", output)
    return int(output)
'''


'''

def choose_tool(state, question, tools_left): 
    tools = list(enumerate(state.tools))

    def makeQuestion(question, tools_to_use):
        return f"\n<Q>{question}</Q>" \
               + "<TASK>Which tool in ["+", ".join([str(i) for i in tools_to_use]) + "] is best to try first?</TASK>" 

    examples = []
    for tool_id, tool in tools:
        for tool_example in tool['examples']:
            examples += [makeQuestion(tool_example, range(len(tools))) + f"<ID>{str(tool_id)}</ID>"]

    random_fixed_seed.shuffle(examples)

    tool_context = "".join([state.makeToolDesc(t) for t,_ in tools])
    prompt = tool_context + "<OBJECTIVE>Decide which available tool is best to answer Q.  Always provides an answer. Uses tool 0 or 1 if the other tools don't fit well.</OBJECTIVE>\n\n" \
           + "".join(examples) \
           + makeQuestion(question, tools_left) + "<ID>"
    print_op(prompt)

    output = call_gpt(state, prompt, "</ID>", max_tokens = 40, temperature = 0).strip()

    if state.verbose > 0:
        print_op("OUTPUT:", output)
    return int(output)



'''
