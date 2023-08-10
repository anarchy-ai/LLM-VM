import openai
from llm_vm.client import Client
import pickle
from math import sqrt, pow, exp
import spacy 
nlp=spacy.load("en_core_web_md")

def squared_sum(x):
    """return 3 rounded square rooted value"""

    return round(sqrt(sum([a * a for a in x])), 3)

def cos_similarity(x, y):
    """return cosine similarity between two lists"""

    numerator = sum(a * b for a, b in zip(x, y))
    denominator = squared_sum(x) * squared_sum(y)
    return round(numerator / float(denominator), 3)

new_file = open("data_gen_e.pkl","rb")
examples = list(pickle.load(new_file))
client = Client(big_model='llama')


client1 = Client(big_model='pythia')

sims = []
for i in examples:
    response_llama = client.complete(prompt = i[0], context = 'Split the "Q" into its subtasks and return that as a list separated by commas. Return an empty string if no subtasks are necessary. \n\n Q: Find all the files in my system that were sent to HR before July 2nd. \n\n Find all files in system., Using previous answer search for files that were sent to HR., Using previous answer search for all files that were sent before July 2nd.<ENDOFLIST>\n\n')["completion"].split("<ENDOFLIST>")[0]
    print("llama:",response_llama)
    response_pythia = client1.complete(prompt = i[0], context = 'Split the "Q" into its subtasks and return that as a list separated by commas. Return an empty string if no subtasks are necessary. \n\n Q: Find all the files in my system that were sent to HR before July 2nd. \n\n Find all files in system., Using previous answer search for files that were sent to HR., Using previous answer search for all files that were sent before July 2nd.<ENDOFLIST>\n\n')["completion"].split("<ENDOFLIST>")[0]
    print("pythia",response_pythia)
    response_chat_gpt = i[1].split("<ENDOFLIST>")[0]
    print(response_chat_gpt)
    final_sims = [cos_similarity(nlp(response_llama).vector, nlp(response_chat_gpt).vector),cos_similarity(nlp(response_pythia).vector, nlp(response_chat_gpt).vector)]
    sims.append(final_sims)
print(sims)
