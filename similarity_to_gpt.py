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

new_file = open("data_gen_eval.pkl","rb")
examples = list(pickle.load(new_file))
sims = []
for i in examples[0:30]:
    client = Client(big_model='llama')
    # specify the file name of the finetuned model to load
    model_name = '2023-08-08T22:22:06_open_llama_3b_v2.pt'
    client.load_finetune(model_name)
    response_llama = client.complete(prompt = i[0], context = '')["completion"].split("<ENDOFLIST>")[0]
    print(response_llama)
    client1 = Client(big_model='pythia')
    # specify the file name of the finetuned model to load
    model_name = '2023-08-08T22:44:34_pythia-70m-deduped.pt'
    client1.load_finetune(model_name)
    response_pythia = client1.complete(prompt = i[0], context = '')["completion"].split("<ENDOFLIST>")[0]
    print(response_pythia)
    response_chat_gpt = i[1].split("<ENDOFLIST>")[0]
    print(response_chat_gpt)
    final_sims = [cos_similarity(nlp(response_llama).vector, nlp(response_chat_gpt).vector),cos_similarity(nlp(response_pythia).vector, nlp(response_chat_gpt).vector)]
    sims.append(final_sims)
print(sims)
