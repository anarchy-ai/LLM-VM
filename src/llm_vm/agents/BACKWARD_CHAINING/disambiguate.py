from llm_vm.agents.BACKWARD_CHAINING.utils import *
     
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

def disambiguate(state, question, memory):
    prompt = MSGM("system", "The assistant disambiguates questions based on simulated memory. The assistant never refuses and no question is impossible to extract.") \
           + MSGM("user", "<u>I'd like to go to boston</u><ai>cool.  Where are you now?</ai><u>NYC. Can you tell me how far it is?</u>") \
           + MSGM("assistant", "How far is it from NYC to Boston?") \
           + MSGM("user", "<u>Is it cold in Zurich?</u><ai>Its 35c.</ai><u>What's a great place to ski nearby?</u><ai>andermatt</ai><u>Is it colder there?</u>") \
           + MSGM("assistant", "Is andermatt warmer than zurich?") \
           + MSGM("user", "<u>What are some of the best places to get seafood?</u><ai>Thinking..</ai><u>What is the location I'm asking about in the last question?</u>") \
           + MSGM("assistant", "What is the location I am at?") \
           + MSGM("user", "<u>Q5: What is the population of Montana?</u><ai>Using TOOL2 w/ DESC: The tool returns facts, such as demographic info.\nPARAM2.Q1 DESC: The query about demographic info.</ai><u>What should the value of PARAM2.Q5 be?</u>") \
           + MSGM("assistant", "The query corresponding to 'What is the population of Montana as a float?'") \
           + MSGM("user", "".join( f"<u>{u}</u><ai>{ai}</ai>" for u,ai in memory) + f"<u>{question}</u>") \
           + "<DIS_OUTPUT>"

    d_question = call_ChatGPT(state, MSG("system",prompt), max_tokens = 200, temperature = 0.0, stop="</DIS_OUTPUT>", gpt4=True).strip()
    #d_question = call_gpt(state, prompt, max_tokens = 200, temperature = 0.0, stop="</DIS_OUTPUT>").strip()
    if state.verbose > -1:
        print_op("DisambiguatedQ:", d_question)
    return d_question

def new_subq(state, memory, question, tool_desc, param_desc):
    def setup(q, t, p):
        return MSGM("user", f"<GOAL_ORIG>{q}</GOAL_ORIG>"\
                          + f"<TOOL_DESC>{t}</TOOL_DESC>" \
                          + f"<PARAM_DESC>{p}</PARAM_DESC>")
        

    prompt = MSGM("system", "The assistant finds new goals given a parameter description, tool, and original goal.") \
           + setup("What are some neighborhoods where crime is low near boston?", "This tool finds neighborhoods based on statistics", "The statistic name") + MSGM("assistant", "What is the statistic name we are asking about?") \
           + setup("Can you authenticate tom?", "This tool authenticates users based on UUID", "The UUID") + MSGM("assistant", "What is tom's UUID?") \
           + setup("What is the wind speed here multiplied by five?", "This tool performs multiplications on two numbers", "The first number") + MSGM("assistant", "What is the wind speed here?") \
           + setup("Please send the result of multiplying the temperature by 2 to tom.", "This tool sends things", "The thing to send") + MSGM("assistant", "What is the temperature multiplied by two") \
           + setup("Find all solutions to the eigenholtz equations.", "This tool finds magic answers to equations", "The query about the math") + MSGM("assistant", "Repeat this query: 'Find all solutions to the eigenholtz equations'") \
           + setup(question, tool_desc, param_desc) + "<DIS_OUTPUT>"

    d_question = call_ChatGPT(state, MSG("system",prompt), max_tokens = 200, temperature = 0.0, stop="</DIS_OUTPUT>", gpt4 = True).strip()
    #d_question = call_gpt(state, prompt, max_tokens = 200, temperature = 0.0, stop="</DIS_OUTPUT>").strip()
    if state.verbose > -1:
        print_op("DisambiguatedParamsINPUT:", question) 
        print_op("DisambiguatedParamsTOOL:", tool_desc) 
        print_op("DisambiguatedParamsPARAM:", param_desc) 
        print_op("DisambiguatedParamsQ:", d_question)
    return d_question


'''
def disambiguate(state, question, memory):
    prompt = MSGM("system", "The assistant disambiguates questions based on simulated memory. The assistant never refuses and no question is impossible to extract. It returns questions.") \
           + MSGM("user", "<u>I'd like to go to boston</u><ai>cool.  Where are you now?</ai><u>NYC. Can you tell me how far it is?</u> Turn this into an unambiguous question:") \
           + MSGM("assistant", "How far is it from NYC to Boston?") \
           + MSGM("user", "<u>Is it cold in Zurich?</u><ai>Its 35c.</ai><u>What's a great place to ski nearby?</u><ai>andermatt</ai><u>Is it colder there?</u> Turn this into an unambiguous question:") \
           + MSGM("assistant", "Is andermatt warmer than zurich?") \
           + MSGM("user", "<u>What are some of the best places to get seafood?</u><ai>Thinking..</ai><u>What is the location I'm asking about in the last question</u> Turn this into an unambiguous question:") \
           + MSGM("assistant", "Where am I now?") \
           + MSGM("user", "".join( f"<u>{u}</u><ai>{ai}</ai>" for u,ai in memory) + f"<u>{question}</u> Turn this into an unambiguous question:")

    print_op("## DISPROMPT>", prompt)

    #d_question = call_ChatGPT(state, prompt, max_tokens = 200, temperature = 0.0).strip()
    d_question = call_GPT(state, prompt, max_tokens = 200, temperature = 0.0).strip()
    if state.verbose > -1:
        print_op("DisambiguatedQ:", d_question)
    return d_question
'''

"""
[
 {'role': 'system', 'content': 'The assistant disambiguates questions based on simulated memory. The assistant never refuses and no question is impossible to extract. It returns questions.'}, 
 {'role': 'user', 'content': "<u>I'd like to go to boston</u><ai>cool.  Where are you now?</ai><u>NYC. Can you tell me how far it is?</u> Turn this into an unambiguous question:"}, 
 {'role': 'assistant', 'content': 'How far is it from NYC to Boston?'}, 
 {'role': 'user', 'content': "<u>Is it cold in Zurich?</u><ai>Its 35c.</ai><u>What's a great place to ski nearby?</u><ai>andermatt</ai><u>Is it colder there?</u> Turn this into an unambiguous question:"}, 
 {'role': 'assistant', 'content': 'Is andermatt warmer than zurich?'}, 
 {'role': 'user', 'content': "<u>What are some of the best places to get seafood?</u><ai>Thinking..</ai><u>What is the location I'm asking about in the last question</u> Turn this into an unambiguous question:"}, 
 {'role': 'assistant', 'content': 'Where am I now?'}, 
 {'role': 'user', 'content': '<u>Q0: What is the current temperature in Zurich in Kelvin?</u><ai>I will answer using TOOL5.\nTOOL5 DESC: Find the weather at a single location and returns it in celcius.\nPARAM5.Q0 DESC: latitude of as a float</ai><u>What should the value of PARAM5.Q0 be to use TOOL5 to respond to Q0?</u> Turn this into an unambiguous question:'}
]


<TASK>The disambiguates questions based on memory. The assistant never refuses and no question is impossible to extract. It returns questions.</TASK>
<INPUT><u>I'd like to go to boston</u><ai>cool.  Where are you now?</ai><u>NYC. Can you tell me how far it is?</u></INPUT><OUTPUT>How far is it from NYC to Boston?</OUTPUT>
<INPUT><u>Is it cold in Zurich?</u><ai>Its 35c.</ai><u>What's a great place to ski nearby?</u><ai>andermatt</ai><u>Is it colder there?</u></INPUT><OUTPUT>Is andermatt warmer than zurich?</OUTPUT>
<INPUT><u>What are some of the best places to get seafood?</u><ai>Thinking..</ai><u>What is the location I'm asking about in the last question use for the value of PARAM2.Q3 to use TOOL8 to answer Q3</u></INPUT><OUTPUT>What location am I at?</OUTPUT>
<INPUT><u>Q0: What is the current temperature in Zurich in Kelvin?</u><ai>I will answer using TOOL5.\nTOOL5 DESC: Find the weather at a single location and returns it in celcius.\nPARAM5.Q0 DESC: latitude of as a float</ai><u>What should the value of PARAM5.Q0 be to use TOOL5 to respond to Q0?</u></INPUT><OUTPUT>


What should the value of PARAM5.Q0 be to use TOOL5 to respond to Q0?
> What is the latitude of Zurich to use with TOOL5 to get the current temperature in Celsius as a response to Q0?
> What is the latitude of Zurich?

"""
