import json
import openai
from agent import Agent
import time
import os, sys

google_tool = {
               'description': "Returns the result of a google search result about information on the internet. Good for retrieving information about the world and live information.",
               'dynamic_params': {"q": 'The natural language input query'},
               'method': 'GET',
               'args': {'url': "https://www.googleapis.com/customsearch/v1",
                         'params': {'key': "Enter Key Here",
                                    'cx' : 'Enter CX Here',
                                    'q': '{q}'}
                        }
               }

class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr

openai.api_key = ""
agent = Agent(openai.api_key,[], verbose=-1)
agent.set_tools([google_tool])

f = open("compositional_celebrities.json")
data = json.load(f)
category = []
q_and_a={}
for i in data["data"]:
    if i["category"] in ['birthplace_rounded_lat', 'birthplace_rounded_lng', 'birthplace_tld', 'birthplace_ccn3', 'birthplace_currency', 'birthplace_currency_short', 'birthplace_currency_symbol', 'birthplace_jpn_common_name', 'birthplace_spa_common_name', 'birthplace_rus_common_name', 'birthplace_est_common_name', 'birthplace_urd_common_name', 'birthplace_callingcode', 'birthyear_nobelLiterature', 'birthdate_uspresident', 'birthyear_masterchamp']:
        if i["category"] not in category:
            category.append(i["category"])
            q_and_a.update({i["category"]:[(i["Question"],i["Answer"])]})
        else:
            q_and_a[i["category"]].append((i["Question"],i["Answer"]))

op = ['birthplace_rounded_lat', 'birthplace_rounded_lng', 'birthplace_tld', 'birthplace_ccn3', 'birthplace_currency', 'birthplace_currency_short', 'birthplace_currency_symbol', 'birthplace_jpn_common_name', 'birthplace_spa_common_name', 'birthplace_rus_common_name', 'birthplace_est_common_name', 'birthplace_urd_common_name', 'birthplace_callingcode', 'birthyear_nobelLiterature', 'birthdate_uspresident', 'birthyear_masterchamp']
for j in range(0,len(op)):
    print(j,op[j], file=sys.stderr)
idx=int(input())

choice=op[idx]
for i in range(100):
    with suppress_output(suppress_stdout=True, suppress_stderr=True):
        try:
            a=agent.run(q_and_a[choice][i][0],[])
        except Exception as e:
            a = [e,""]
    print(str(i)+"|",str(a[0])+"|",q_and_a[choice][i][1], file=sys.stderr)
