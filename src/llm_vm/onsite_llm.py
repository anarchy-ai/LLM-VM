import abc
from abc import ABC,abstractmethod
import openai
import math
from transformers import AutoTokenizer, OPTForCausalLM,BloomForCausalLM,LlamaTokenizer, LlamaForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer, DataCollatorForLanguageModeling,TrainingArguments, Trainer
import time
import tempfile
import json
import torch

def create_jsonl_file(data_list):
    out = tempfile.TemporaryFile('w+')
    for a,b in data_list:
        out.write(json.dumps({'prompt': a, 'completion': b}) + "\n")
    out.seek(0)
    return out


class FinetuningDataset(torch.utils.data.Dataset):
    def __init__(self,iterable_dataset,length):
        self.dataset = list(iterable_dataset)   
        self.length = length
    def __len__(self):
        print("length",str(self.length))
        return self.length
    def __getitem__(self, idx):
        print("index",str(idx))
        print(self.dataset)
        return self.dataset[idx]
    
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

    # def generate() # this is where the meat and potatoes should live?

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
    def finetune(self,data, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        final_dataset = []
        for i in data:
            final_dataset =  i[0] + i[1]
        f_dataset = map(self.tokenizer,final_dataset)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="Neo_finetuned",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=2,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            eval_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            data_collator=data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss'])

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
    def finetune(self,data, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        final_dataset = []
        for i in data:
            final_dataset =  i[0] + i[1]
        f_dataset = map(self.tokenizer,final_dataset)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="Neo_finetuned",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=2,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            eval_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            data_collator=data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss'])

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
    
    def finetune(self,data, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        untokenized_final_dataset = []
        for prompt,response in data:
            untokenized_final_dataset.append(prompt + response)
        #    = map(untokenized_final_dataset, lambda x : x[0]+x[1])
        tokenized_final_dataset = map(self.tokenizer,untokenized_final_dataset)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="Neo_finetuned",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size = 1,
            per_device_eval_batch_size = 1,
            num_train_epochs=1,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset)),
            eval_dataset=FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset)),
            data_collator=data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss']) #perplexity is the metric we use for finetuning measurement
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
    def finetune(self,data, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        final_dataset = []
        for i in data:
            final_dataset =  i[0] + i[1]
        f_dataset = map(self.tokenizer,final_dataset)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="Neo_finetuned",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=2,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            eval_dataset=FinetuningDataset(f_dataset,len(final_dataset)),
            data_collator=data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        return math.exp(eval_results['eval_loss'])

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
        ans = openai.Completion.create(prompt= prompt, model="text-davinci-003", **kwargs)
        return ans['choices'][0]['text']
    
    
    def finetune(self, dataset, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        training_file = create_jsonl_file(dataset)
        upload_response = openai.File.create(file=training_file, purpose="fine-tune")
        training_file.close()
        fine_tuning_job = openai.FineTune.create(training_file= upload_response.id)

        print(f"Fine-tuning job created: {fine_tuning_job}", flush=True)
        global job_id # global state isn't great, but thats interrupt handlers
        job_id = fine_tuning_job["id"]
        while True:
            fine_tuning_status = openai.FineTune.retrieve(id=job_id)
            status = fine_tuning_status["status"]
            print(f"Fine-tuning job status: {status}")
            if status in ["succeeded", "completed", "failed"]:
                break
            time.sleep(30)
        job_id = None #
        new_model_id = fine_tuning_status.fine_tuned_model

        print("New_model_id: ", new_model_id, flush=True)

        optimizer.storage.set_model(c_id, new_model_id)
        optimizer.storage.set_training_in_progress(c_id, False)
        if old_model is not None:
            openai.Model.delete(old_model)

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

    def finetune(self, dataset, optimizer, c_id):
        old_model = optimizer.storage.get_model(c_id)
        training_file = create_jsonl_file(dataset)
        upload_response = openai.File.create(file=training_file, purpose="fine-tune")
        training_file.close()
        fine_tuning_job = openai.FineTune.create(training_file= upload_response.id)

        print(f"Fine-tuning job created: {fine_tuning_job}", flush=True)
        global job_id # global state isn't great, but thats interrupt handlers
        job_id = fine_tuning_job["id"]
        while True:
            fine_tuning_status = openai.FineTune.retrieve(id=job_id)
            status = fine_tuning_status["status"]
            print(f"Fine-tuning job status: {status}")
            if status in ["succeeded", "completed", "failed"]:
                break
            time.sleep(30)
        job_id = None #
        new_model_id = fine_tuning_status.fine_tuned_model

        print("New_model_id: ", new_model_id, flush=True)

        optimizer.storage.set_model(c_id, new_model_id)
        optimizer.storage.set_training_in_progress(c_id, False)
        if old_model is not None:
            openai.Model.delete(old_model)