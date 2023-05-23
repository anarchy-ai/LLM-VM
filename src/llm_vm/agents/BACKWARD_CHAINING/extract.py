try:    
    from .utils import *
except:
     from utils import *
     
import string
import re
import sys
import random

def MSGM(u, c):
    #return [{'role': u, 'content' : c}]
    if u == "system":
        return f"<TASK>{c}</TASK>"
    if u == "user":
        return f"<DIS_INPUT>{c}</DIS_INPUT>"
    if u == "assistant":
        return f"<DIS_OUTPUT>{c}</DIS_OUTPUT>"



def extract(state, q, tool_input):
    prompt = MSGM("system", "The assistant extracts just the answer. It never refuses.") \
           + MSGM("user", "Q: bucket\nPRE-EXTRACTION: The bucket for the question about distance between aliens and robots is drunk cats") \
           + MSGM("assistant", "drunk cats") \
           + MSGM("user", "Q: time in 24h format\nPRE-EXTRACTION: 5:30 is the beginning time for the start of the movie") \
           + MSGM("assistant", "17:30") \
           + MSGM("user", "Q: an integer area in m^2\nPRE-EXTRACTION: the area of the circle is 1234^2, which is usefull for sushi") \
           + MSGM("assistant", "1234") \
           + MSGM("user", f"Q: {q}\nPRE-EXTRACTION: {tool_input}") + "<DIS_OUTPUT>"

    if state.verbose > -1:
        print_op("TOOL_INPUT_ORIG: ", tool_input)
    extracted = call_ChatGPT(state, MSG("system",prompt), max_tokens = 200, temperature = 0.0, stop = "</DIS_OUTPUT>", gpt4=True).strip()
    if state.verbose > -1:
        print_op("ExtractedQ:", extracted)
    return extracted.strip()

'''
def extract(state, q, tool_input):
    prompt = MSG("system", "The assistant extracts just the answer. It never refuses.") \
           + MSG("user", "Q: bucket\nEXTRACT ANSWER: The bucket for the question about distance between aliens and robots is drunk cats") \
           + MSG("assistant", "drunk cats") \
           + MSG("user", "Q: time5\nEXTRACT ANSWER: time5=5:30 is the beginning time for the start of the movie") \
           + MSG("assistant", "5:30") \
           + MSG("user", "Q: PARAM123.quack0\nEXTRACT ANSWER: PARAM123.quack0 = a duck and a goat and PARAM123.quack1 is zones") \
           + MSG("assistant", "duck and a goat") \
           + MSG("user", f"Q: {q}\nEXTRACT ANSWER: {tool_input}")

    if state.verbose > -1:
        print_op("TOOL_INPUT_ORIG: ", tool_input)
    extracted = call_ChatGPT(state, prompt, max_tokens = 200, temperature = 0.0).strip()
    if state.verbose > -1:
        print_op("ExtractedQ:", extracted)
    return extracted.strip()

'''

'''

def extract(state, q, tool_input):
    cur_prompt = "" \
        +  "<LONG-ANS>The bucket for the question about distance between aliens and robots is drunk cats</LONG-ANS><SHORT-ANS>drunk cats</SHORT-ANS>\n" \
        +  "<LONG-ANS>5:30 is the beginning time for the start of the movie</LONG-ANS><SHORT-ANS>5:30</SHORT-ANS>\n" \
        + f"<LONG-ANS>{tool_input}</LONG-ANS><SHORT-ANS>"

    print_op("TOOL_INPUT_ORIG: ", tool_input)

    extracted = call_gpt(state, cur_prompt, f"</SHORT-ANS>", max_tokens = 100,
                         quality = "okay", 
                         temperature = 0.0).strip()
    if state.verbose > -1:
        print_op("ExtractedQ:", extracted)
    return extracted.strip()

'''
