import abc
from abc import ABC,abstractmethod
import openai
from transformers import AutoTokenizer, OPTForCausalLM,BloomForCausalLM,LlamaTokenizer, LlamaForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer

class Base_Onsite_LLM(ABC):
    def __init__(self,model_uri_override=None,tokenizer_kw_args=None,model_kw_args=None):
        if model_uri_override != None:
            self.model_uri= model_uri_override 
        self.model=model_loader()
 
    @property
    def model_uri(self):
        pass

    @model_uri.setter
    def model_uri(self,val):
        self.model_uri=val # check if this is correct

    @abstractmethod
    def model_loader(self):
        pass

    @abstractmethod
    def tokenizer_loader(self):
        pass

    @abstractmethod
    def generate(self): # this is where the meat and potatoes should live?
        pass

"""
this factorization isn't necessarily the greatest, nor should it be viewed
as likely being more general, aside from covering hugging face transformers
"""

class Small_Local_OPT:
    
    """
    This is a class for Facebook's OPT-350m LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    
    def __init__(self,model_uri_override="facebook/opt-350m"): # tokenizer_kw_args=None,model_kw_args=None
        self.model_uri = model_uri_override
        self.tokenizer=self.tokenizer_loader()
        self.model= self.model_loader()

    def model_loader(self):
        return OPTForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)
    def generate(self,prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses the class's llm and tokenizer to generate a response given a user's prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
           >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
           I think it takes about a week for the apple to grow.
        """
        inputs=self.tokenizer(prompt,return_tensors="pt")
        generate_ids=self.model.generate(inputs.input_ids,max_length=max_length)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally 
        return resp[len(prompt):]

class Small_Local_Bloom:

    """
    This is a class for BigScience's bloom-560 LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    def __init__(self,model_uri_override="bigscience/bloom-560m"): # tokenizer_kw_args=None,model_kw_args=None
        self.model_uri = model_uri_override
        self.tokenizer=self.tokenizer_loader()
        self.model= self.model_loader()

    def model_loader(self):
        return BloomForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)
    def generate(self,prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses the class's llm and tokenizer to generate a response given a user's prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
           >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
            How long does it take for a tomato...
        """
        inputs=self.tokenizer(prompt,return_tensors="pt")
        generate_ids=self.model.generate(inputs.input_ids,max_length=max_length)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally 
        return resp[len(prompt):]

 #

class Small_Local_Neo:

    """
    This is a class for BigScience's bloom-560 LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    def __init__(self,model_uri_override="EleutherAI/gpt-neo-1.3B"): # tokenizer_kw_args=None,model_kw_args=None
        self.model_uri = model_uri_override
        self.tokenizer=self.tokenizer_loader()
        self.model= self.model_loader()

    def model_loader(self):
        return GPTNeoForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return GPT2Tokenizer.from_pretrained(self.model_uri)
    def generate(self,prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses the class's llm and tokenizer to generate a response given a user's prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
           >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
           The apple tree is a very slow growing plant...
        """
        inputs=self.tokenizer(prompt,return_tensors="pt")
        generate_ids=self.model.generate(inputs.input_ids,max_length=max_length)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally 
        return resp[len(prompt):]
#
class Small_Local_LLama:

    """
    This is a class for Openlm-Research's open_llama-3b LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    def __init__(self,model_uri_override="openlm-research/open_llama_3b"): # tokenizer_kw_args=None,model_kw_args=None
        self.model_uri = model_uri_override
        self.tokenizer=self.tokenizer_loader()
        self.model= self.model_loader()

    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)
    def generate(self,prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses the class's llm and tokenizer to generate a response given a user's prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
           >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
           How long does it take for an apple tree to grow?
        """
        inputs=self.tokenizer(prompt,return_tensors="pt")
        # the example calls for max_new_tokens
        generate_ids=self.model.generate(inputs.input_ids,max_length=max_length)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally 
        return resp[len(prompt):]

class GPT3:

    """
    This is a class for openAI's completion endpoint

    Methods:
        generate: Generates a response from a given prompt with OpenAI's completion endpoint
    """
    
    def generate(self,prompt, max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses openAI's API to generate a response from the prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
            >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
            It typically takes about 100-200 days...
        """
        ans = openai.Completion.create(prompt= prompt, model='text-davinci-003', **kwargs)
        return ans['choices'][0]['text']

class Chat_GPT:
    """
    This is a class for openAI's gpt-3.5-turbo LLM

    Methods:
        generate: Generates a response from a given prompt through OpenAI's endpoint
    """
    
    def generate(self,prompt, max_length=100,**kwargs): # both tokenizer and model take kwargs :(
        """
        This function uses openAI's API to generate a response from the prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response
        
        Example:
            >>> Small_Local_OPT.generate("How long does it take for an apple to grow?)
            It typically takes about 100-200 days...
        """
        cur_prompt = [{'role': "system", 'content' : prompt}]
        ans = openai.ChatCompletion.create(
            messages=cur_prompt,
            model="gpt-3.5-turbo-0301",
            **kwargs)
        return ans['choices'][0]['message']['content']

