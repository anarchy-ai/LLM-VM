"""
This Python script runs the REBEL agent, which takes into account of multiple questions within a 
single user input. 
The user is prompted to enter a question, to which the AI responds. 
If multiple questions are present, the script splits these into different subquestions and retrieves 
answers from the respective API.  
The script also takes into account of previous prompts as context, in case the user may 
enter related questions later.  
The user will also see messages regarding which API for information was chosen and the 
price of the API call.  
"""

import os
import sys

# Get the current file's directory to grab the python files with common functionality in the utils/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(grandparent_dir, 'utils/')
sys.path.append(utils_dir)

from keys import * 
from labels import *
from print_types import * 
from tools import * 
from urllib.parse import urlencode
import urllib.parse as urlparse
import json

import random

# UNCOMMENT THESE IMPORTS SO THAT SUBQUESTION PORTION CAN WORK (NLP CURRENTLY NOT RECOGNIZED)
# import spacy
# nlp =  spacy.load("en_core_web_md")

from math import sqrt, pow, exp
try:
    from .bothandler import question_split,tool_picker,memory_check, replace_variables_for_values
    from .utils import * 
except:
    from bothandler import question_split,tool_picker,memory_check, replace_variables_for_values
    from utils import * 



def prepPrintPromptContext(p):
    return "--> "+p.replace("\n", "\n--> ") if len(p) > 0 else ""

def print_op(*kargs, **kwargs):
    print(*kargs, **kwargs, flush=True)

random_fixed_seed = random.Random(4)

QUALITY = 0.2

def buildGenericTools(tools=GENERIC_TOOLS):

    new_tools = tools

    # wolfram 
    new_tools[0]['examples'] = [([("What's the most popular spot to vacation if you are in Germany?", "Mallorca"),
                              ("What's the date", "July 7, 2009"),
                              ], "How crowded is it there now?", '{"q": "How many tourists visit Mallorca each July?"}'),

                            ([("What is the circumference of a basketball?", "750mm"),
                              ], "What is the volume of a basketball?", '{"q": "What is the volume of a ball with circumference 750mm?"}'),

                            ([("What's the fastest a ford explorer can drive?", "160mph"),
                              ], "What is that multiplied by seventy?", '{"q": "160 x 70"}'),

                            ([], "Is 5073 raised to the 3rd power divisible by 73?",
                             '{"q": "Is 5073^3 divisible by 73?"}')
                        ]
    
    # geopy
    new_tools[1]['examples'] = [([("I feel like taking a drive today to zurich can you help?", "Yes!  Where are you now?"),
                              ], "I'm in Paris.", '{"origins": "Paris", "destinations": "Zurich"}'),
                            ([], "How long would it take to get between South Africa and Kenya.",
                             '{"origins": "South Africa", "destinations": "Kenya"}'),
                            ([("Where do elephant seals mate?", "San Luis Obispo"),
                              ("Thats really cool!", "Damn right."),
                              ("Where do they migrate each year?", "Alaska"),
                              ], "How many miles would they travel while doing that?", '{"origins": "San Luis Obispo", "destinations": "Alaska"}')
                            ]
    
    # weather 
    new_tools[2]['examples'] = [([("Are you allergic to seafood?", "I'm just an AI. I have no body."),
                              ("What city is Obama in?", "NYC"),
                              ("What is the latitude of NYC?", "40.7128° N"),
                              ("What is the longitude of NYC?",
                               "The longitude is 73° 56' 6.8712'' W"),
                              ], "What is the weather in NYC?", '{"longitude": 73.56, "latitude": 40.41728}'),
                            ([("What is the longitude of Milan?", "9.1900° E"),
                              ("What is the latitude of Milan?", "45.4642° N"),
                              ], "What is the weather in Milan?", '{"longitude": 45.4642, "latitude": 9.1900}')
                            ]
    return new_tools
 
def squared_sum(x):
  """ return 3 rounded square rooted value """
 
  return round(sqrt(sum([a*a for a in x])),3)

def cos_similarity(x,y):
  """ return cosine similarity between two lists """
 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

class Agent:
    def __init__(self, openai_key, tools, bot_str="", verbose = 4):
        self.verbose = verbose
        self.price=0
        self.set_tools(tools)
        if bot_str=="":
            self.bot_str=bot_str
        else:
            self.bot_str = "<GLOBAL>"+bot_str+"<GLOBAL>"

        # set all the API resource keys to make calls 
        set_api_key(openai_key,"OPENAI_API_KEY")
        set_api_key(GOOGLE_MAPS_KEY, "GOOGLE_MAPS_KEY")
        set_api_key(SERPAPI_KEY, "SERPAPI_KEY")
        set_api_key(WOLFRAM_KEY, "WOLFRAM_KEY")

    def makeToolDesc(self, tool_id):
        tool = self.tools[tool_id]
        params = "{"+", ".join(['"'+l + '": '+v for l, v in tool['dynamic_params'].items()]
                               )+"}" if tool['dynamic_params'] != "" else "{}"
        return f'''<TOOL>
<{TOOL_ID}>{str(tool_id)}</{TOOL_ID}>
<{DESCRIPTION}>{tool['description']}</{DESCRIPTION}>
<{PARAMS}>{params}</{PARAMS}>
</TOOL>'''

    def set_tools(self, tools):
        self.tools = []
        for tool in tools:

            if not 'args' in tool:
                tool['args'] = {}
            if not 'method' in tool:
                tool['method'] = "GET"
            if not 'examples' in tool:
                tool['examples'] = []
            if not 'dynamic_params' in tool:
                tool['dynamic_params'] = {}
            self.tools += [tool]
        


    def use_tool(self, tool, gpt_suggested_input, question, memory, facts, query=""):

        if query=="":
            query=question
       
        if gpt_suggested_input[0] != "{":
            gpt_suggested_input = "{"+gpt_suggested_input
        if gpt_suggested_input[-1] != "}":
            gpt_suggested_input += "}"

        if self.verbose > 1:
            print_op("GPT SUGGESTED INPUT:", gpt_suggested_input)
        
        parsed_gpt_suggested_input = json.loads(gpt_suggested_input)
            
    
        
        #make sure all of the suggested fields exist in the tool desc
        for i in parsed_gpt_suggested_input.keys():
                if i not in tool["dynamic_params"].keys():
                    raise Exception("Bad Generated Input")
        tool_args = replace_variables_for_values(tool["args"], parsed_gpt_suggested_input)
        
        url = tool_args['url']
        if self.verbose > -1:
            print_op(tool['method']+":", url)

        if 'auth' in tool_args and isinstance(tool_args['auth'], dict):
            auths = list(tool_args['auth'].items())
            if len(auths) > 0:
                tool_args['auth'] = list(tool_args['auth'].items())[0]
            else:
                del tool_args['auth']
              
        # Remove those parameters that are not part of Python's `requests` function.
        tool_args.pop("jsonParams", None)
        tool_args.pop("urlParams", None)
        
        if self.verbose > -1:
            print_op("ARGS: ", tool_args)

        resp = (requests.get if tool['method'] ==
                'GET' else requests.post)(**tool_args)
        
        #print_op("FINAL URL: (" + tool["method"] + ") ", resp.url)

        actual_call = str(tool_args)

        if resp.status_code in [404, 401, 500]:
            er = " => "+ str(resp.status_code)
            return "This tool isn't working currently" + er

        ret = str(resp.text)
        if self.verbose > 4:
            print(ret)
        try:
            ret = str(json.loads(ret))
        except:
            pass

        if len(ret) > 10000:
            ret = ret[0:10000]
        mem = "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "Human-AI") for p,a in memory]) \
                    + "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "AI-AI") for p,a in facts])
       
        prompt=MSG("system","You are a good and helpful bot"+self.bot_str)
        prompt+=MSG("user",mem+"\nQ:"+query+"\n An api call about Q returned:\n"+ret+"\nUsing this information, what is the answer to Q?")
        a = call_ChatGPT(self, prompt, stop="</AI>", max_tokens = 256).strip()
        return a

    def run(self, question, memory):

        self.price = 0

        thought, facts = self.promptf(question, memory, [],)

    
        if self.verbose > -1:
            print_big("GPT-3.5 Price = ~{:.1f} cents".format(self.price * 100))

        # return (answer, memory + [(question, answer)], calls, debug_return, has_friendly_tags)
        return (thought, memory + [(question, thought)])

    def makeInteraction(self, p,a, Q = "HUMAN", A = "AI", INTERACTION = INTERACTION):
        return f"<>{INTERACTION}:<{Q}>{p}</{Q}>\n<{A}>"+(f"{a}</{A}>\n</>" if a is not None else "")

    def make_sub(self, tools, memory, facts, question, subq, answer_label, toolEx, tool_to_use = None, bot_str = "", quality= "best", max_tokens = 20):
        tool_context = "".join([self.makeToolDesc(t) for t,_ in tools])
        mem = "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "Human-AI") for p,a in memory]) \
                + "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "AI-AI") for p,a in facts])
        def makeQuestion(memory, question, tool = None):
            
            return f"\n\n<{EXAMPLE}>\n" \
                + f"<{QUESTION}>{question}</{QUESTION}>\n" \
                + f"<{THOUGHT}><{PROMPT}>{subq(tool)}</{PROMPT}>\n<{RESPONSE} ty={answer_label}>\n"

        examples = []
        for tool_id, tool in tools:
            for tool_example in tool['examples']:
                examples += [makeQuestion(tool_example[0], tool_example[1], tool_id) + toolEx(tool_id, tool_example) + f"</{RESPONSE}></{THOUGHT}></{EXAMPLE}>"]

        random_fixed_seed.shuffle(examples)
        
        cur_question = f"<{CONVERSATION}>{mem}</{CONVERSATION}>" + makeQuestion(memory, question, tool = tool_to_use) 
        prompt=MSG("system","You are a good and helpful assistant.")
        prompt+=MSG("user","<TOOLS>"+tool_context + "</TOOLS>" + "".join(examples) + cur_question)
        return call_ChatGPT(self,prompt,stop=f"</{RESPONSE}>", max_tokens = max_tokens).strip()
    
    def promptf(self, question, memory, facts, split_allowed=True, spaces=0):
        for i in range(spaces):
            print(" ",end="")

        mem = "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "Human-AI") for p,a in memory]) \
            + "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "AI-AI") for p,a in facts])
    
        if split_allowed:
            subq=question_split(question,self.tools,mem)
            for i in range(spaces):
                print(" ",end="")
            subq_final=[]
            print(subq[1])
            if len(subq[1])==1:
                split_allowed=False

            else:

                for i in subq[1]:
                    
                    if cos_similarity(nlp(question).vector, nlp(i).vector) < 0.98 :
                        subq_final.append(i)
                    else:    
                        split_allowed = False
                        
            self.price+=subq[0]
            new_facts=[]
            
            for i in range (len(subq_final)):
                
                _, new_facts= self.promptf(subq_final[i], memory, facts, split_allowed=split_allowed, spaces=spaces+4)
                facts=facts+new_facts
                mem = "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "Human-AI") for p,a in memory]) \
                    + "".join([self.makeInteraction(p,a, "P", "AI", INTERACTION = "AI-AI") for p,a in facts])

                
        
        
        
        answer_in_memory=memory_check(mem,question)
        self.price+=answer_in_memory[0]
        answer_in_memory=answer_in_memory[1]
        if answer_in_memory: 
           
            prompt=MSG("system","You are a good and helpful bot"+self.bot_str)
            prompt+=MSG("user",mem+"\nQ:"+question+"\nANSWER Q, DO NOT MAKE UP INFORMATION.")
            a = call_ChatGPT(self, prompt, stop="</AI>", max_tokens = 256).strip()
            print_big("NO DATA TOOLS USED, ANSWERED FROM MEMORY")
            return (a.replace("\n",""), [(question, a)])
           

        tool_to_use=tool_picker(self.tools,question,0)

        # TODO: UNCOMMENT PRINT STATEMENT WHEN SPACY WORKS FOR INTERFACE CONSISTENCY (NEEDS TO BE TESTED)
        # print_big(
        #     "".join([f'{"✓" if self.tools.index(tool) == tool_to_use else " "} {self.tools.index(tool)}.- {tool["description"]}\n' for tool in self.tools]) +
        #     f"\n> Question: {question}\n> Raw answer: '{tool_to_use}'\n> Tool ID: {tool_to_use}", "LIST OF DATA TOOLS"
        # )

        self.price+=tool_to_use[0]
        try:
            tool_to_use=int(tool_to_use[1])
        except:
            tool_to_use=len(self.tools)   
        if tool_to_use==len(self.tools):
            prompt=MSG("system","You are a good and helpful bot"+self.bot_str)
            prompt+=MSG("user",mem+"\nQ:"+question+"\nUsing this information, what is the answer to Q?")
            a = call_ChatGPT(self, prompt, stop="</AI>", max_tokens = 256).strip()

            return (a.replace("\n",""), [(question, a)])
        tool_input = self.make_sub(list(enumerate(self.tools)),
                                    memory, facts, 
                                    question, 
                                    lambda t: "What should the input for tool "+str(t)+" be to answer Q?", 
                                    "JSON",
                                    lambda t,ex: ex[2], 
                                    tool_to_use = tool_to_use, 
                                    quality = "best" if self.price < QUALITY else "okay", 
                                    max_tokens = 200)
        

        query=question
        if "ai_response_prompt" in self.tools[tool_to_use].keys():
            query=self.tools[tool_to_use]["ai_response_prompt"]
        answer = self.use_tool(self.tools[tool_to_use], tool_input, question, memory, facts, query=query)
        for i in range(spaces):
            print(" ",end="")

        return (answer, [(question, answer)])

    def call_gpt(self, cur_prompt: str, stop: str, max_tokens = 20, quality = "best", temperature = 0.0):

        if self.verbose > 2:
            print_op(prepPrintPromptContext(cur_prompt))
            
        ask_tokens = max_tokens + len(cur_prompt) / 2.7

        if self.verbose > 0:
            print_op("ASK_TOKENS:", ask_tokens)

        if (ask_tokens) > 2049:
            quality = 'best'

        model = { 'best' : ("text-davinci-003", 0.02), 
                  'okay' : ("text-curie-001", 0.002), 
                 }[quality]

        def calcCost(p):
            return (len(p) / 2700.0) * model[1]

        try:
            self.price += calcCost(cur_prompt)

            ans = openai.Completion.create(
                model=model[0],
                max_tokens=max_tokens,
                stop=stop,
                prompt=cur_prompt,
                temperature=temperature
            )
        except Exception as e:
            print_op("WTF:", e)
            return "OpenAI is down!"

        response_text = ans['choices'][0]['text']
        self.price += calcCost(response_text)

        if self.verbose > 2:
            print_op("GPT output:")
            print_op(prepPrintPromptContext(response_text))
            print_op("GPT output fin.\n")

        return response_text

# print_op(google(' {"question": ""}'))
if __name__ == "__main__":
    
        '''
        # nytimes
        tools += [{'url': "https://api.nytimes.com/svc/mostpopular/v2/viewed/1.json?api-key=YAKJdTt2RhGZfFNSwEfZTmQmloUGmMjL",
                'description': "use this tool to find the top news stories of the day", 
                'params': [],
                'examples' : []}] 
        '''
        #tools = [{'method': 'GET',"description":"use this tool to find the price of stocks",'args' : {"url":"https://finnhub.io/api/v1/quote",'params': { 'token' :'cfi1v29r01qq9nt1nu4gcfi1v29r01qq9nt1nu50'} },"dynamic_params":{"symbol":"the symbol of the stock"}}]
        
        tools = buildGenericTools()

        tools +=  [{'method': 'GET', "dynamic_params":{ 'location': 'This string indicates the geographic area to be used when searching for businesses. \
        Examples: "New York City", "NYC", "350 5th Ave, New York, NY 10118".', 'term': 'Search term, e.g. "food" or "restaurants". The \
        term may also be the business\'s name, such as "Starbucks"'},"description":"This tool searches for a business on yelp.  It's useful for finding restaurants and \
        whatnot.",'args' :{'url': 'https://api.yelp.com/v3/businesses/search', 'cert': '', 'json': {}, 'params': {'limit': '1', 
        'open_now': 'true', 'location': '{location}', 'term': '{term}'}, 'data': {}, 
        'headers': {'authorization': 'Bearer OaEqVSw9OV6llVnvh9IJo92ZCnseQ9tftnUUVwjYXTNzPxDjxRafYkz99oJKI9WHEwUYkiwULXjoBcLJm7JhHj479Xqv6C0lKVXS7N91ni-nRWpGomaPkZ6Z1T0GZHYx', 
        'accept': 'application/json'}}}]
        
            
        label = Agent(OPENAI_DEFAULT_KEY, tools, verbose=1)
        conversation_history = []
        last = ""
        while True:
            inp = input(last+"Human: ")
            return_value = label.run(inp, conversation_history)
            conversation_history = return_value[1]
            last = "AI: "+str(return_value[0]) + "\n"
