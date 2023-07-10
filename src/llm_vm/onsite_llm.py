# none of these models are of unique special quality,
# merely ones that are easy to load and test on small machines

import abc
from abc import ABC,abstractmethod

from transformers import AutoTokenizer, OPTForCausalLM,BloomForCausalLM,LlamaTokenizer, LlamaForCausalLM

# on a larger machine GPTNeoXForCausalLM, GPTNeoXTokenizerFast

# print("yup")

# something like this interface 
class Base_Onsite_LLM(ABC):
    def __init__(self,model_uri_override=None,tokenizer_kw_args=None,model_kw_args=None):
        if model_uri_override != None:
            self.model_uri= model_uri_override 
        self.model=model_loader()

    # @abstractmethod
    @property
    def model_uri(self):
        pass

    @model_uri.setter
    def model_uri(self,val):
        self.model_uri=val # ummm, is this correct?

    @abstractmethod
    def model_loader(self):
        pass

    @abstractmethod
    def tokenizer_loader(self):
        pass

    # def generate() # this is where the meat and potatoes should live?

"""
this factorization isn't necessarily the greatest, nor should it be viewed
as likely being more general, aside from covering hugging face transformers
"""

class Small_Local_OPT: 

    """
from transformers import AutoTokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

prompt = "Hey, are you conscious? Can you talk to me?"

inputs = tokenizer(prompt, return_tensors="pt")

# Generate

generate_ids = model.generate(inputs.input_ids, max_length=30)

tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"""
    def __init__(self,model_uri_override=None): # tokenizer_kw_args=None,model_kw_args=None
        if (model_uri_override is None): 
                self.model_uri="facebook/opt-350m"
        else: 
                self.model_uri=model_uri_override
        self.tokenizer=self.tokenizer_loader()
        self.model= self.model_loader()

    def model_loader(self):
        return OPTForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)
    def generate(prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :( 
        inputs=self.tokenizer(prompt,return_tensors="pt")
        generate_ids=self.model.generate(self,inputs.input_ids,max_length=max_length)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally 
        return resp[len(prompt):]





# below are the scripts for the other models 

"""
import torch

from transformers import AutoTokenizer,AutoModel, BloomForCausalLM



checkpoint = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto",save_folder="/Users/carter/OFFLOAD`")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
input_ids = inputs.input_ids
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

 """

 # 
"""
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
gen_text[len(prompt):] # might need to trim leading \n 

"""

#

"""
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)

model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map='auto',offload_folder="/Users/carter/OFFLOAD"
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)

print(tokenizer.decode(generation_output[0]))
 """


