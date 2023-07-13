  
from llm_vm.agents.BACKWARD_CHAINING.utils import *
     
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
        return f"<INPUT>{c}</INPUT>"
    if u == "assistant":
        return f"<OUTPUT>{c}</OUTPUT>"



def contained(state, question, memory):
    prompt = MSGM("system", "assistant outputs True if the last request can be completed using only conversational history, and False otherwise.") \
           + MSGM("user", "<u>I'd like to go to botok</u><ai>cool.  Where are you now?</ai><u>Antartica. Can you tell me how far it is?</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>What's USER 12341Aq5's password</u><ai>0349861alsdfa</ai><u>Sorry I mean USER 12341bq5</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>Is it cold in ZulakPol?</u><ai>Its 35c.</ai><u>What's a great colder place to ski nearby?</u><ai>andermatt</ai><u>Is it colder there?</u> Can use history?") \
           + MSGM("assistant", "True") \
           + MSGM("user", "<u>I want to make tea, at what temperature does it boil?</u><ai>100c</ai><u>What about alcohol?</u><ai>Thinking...</ai><u>What is the substance I'm asking about in the prior question?</u> Can use history?") \
           + MSGM("assistant", "True") \
           + MSGM("user", "<u>I want to make tea, at what temperature does it boil?</u><ai>100c</ai><u>What about alcohol?</u><ai>Thinking...</ai><u>What is the boiling temperature of alcohol?</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>How many votes does biden currently have</u><ai>5</ai><u>What about Jill?</u><ai>90</ai><u>How many votes did the person I was asking about have?</u> Can use history?") \
           + MSGM("assistant", "True") \
           + MSGM("user", "<u>What time is it?</u><ai>5:20pm</ai><u>What time is it?</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>Is it cold in PresperKombo?</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>The query about your face</u> Can use history?") \
           + MSGM("assistant", "True") \
           + MSGM("user", "Great job!  These answers have been perfect so far! <u>What's the time?</u><ai>1:00am EST</ai><u>What time did you say it was?</u> Can use history?") \
           + MSGM("assistant", "True") \
           + MSGM("user", "<u>How many votes does biden currently have</u><ai>5</ai><u>What about Jill?</u><ai>90</ai><u>I'm talking about Jill Biden</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "<u>Where are my shoes?</u><ai>In your cupboard</ai><u>What about John's?</u> Can use history?") \
           + MSGM("assistant", "False") \
           + MSGM("user", "".join( f"<u>{u}</u><ai>{ai}</ai>" for u,ai in memory) + f"<u>{question}</u> Can use history?") + "<OUTPUT>"

    can_answer_from_history = call_ChatGPT(state, MSG("system",prompt), stop="</OUTPUT>", max_tokens = 10, temperature = 0.0, gpt4=False).strip()

    return can_answer_from_history == "True"
        

def splitworthy(state, question):
    split_prompts = ["Call my phone and send it a weird text message.",
                     "Compare the distance between NYC and Zurich and some temperature in the arctic.",
                     "What is the rank of Q29. What about B?",
                     "What's the population of Boston first then Bilbao?",
                     "Tell me the wind-speed of the atlantic and pacific"]

    no_split_prompts = ["What's the temperature of Bilbao?",
                        "Tell me how long it'll take to get between Florida and IL by car",
                        "What is the rank of Q29",
                        "What time is it?"]

    
    
    def quest(a):
        return MSG("user", f"<u>{a}</u> Can be split?")

    exs = [ quest(a) + MSG("assistant", "True") for a in split_prompts] \
        + [ quest(a) + MSG("assistant", "False") for a in no_split_prompts]

    random_fixed_seed.shuffle(exs)    

    prompt = MSG("system", "assistant outputs True if the last request should be easily split into two steps, and False otherwise.") \
           + flatten(exs) \
           + quest(question) 

    return call_ChatGPT(state, prompt, max_tokens = 10, temperature = 0.0).strip() == "True"
        
