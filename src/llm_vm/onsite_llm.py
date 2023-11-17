from abc import ABC,abstractmethod
import sys
import openai
import math
from ctransformers import AutoModelForCausalLM
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    OPTForCausalLM,
    BloomForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    CodeLlamaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig)
import time
from datetime import datetime
import tempfile
import json
import os
import torch
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from sentence_transformers import SentenceTransformer



__private_key_value_models_map =  {}
# []   {
#         "opt": SmallLocalOpt,
#         "bloom": SmallLocalBloom,
#         "neo": SmallLocalNeo,
#         "llama2": SmallLocalLLama,
#         "gpt": GPT3,
#         "chat_gpt": ChatGPT,
#         "flan" : SmallLocalFlanT5,
#         "pythia" : SmallLocalPythia,
#         }

# Check available devices
if torch.cuda.device_count() > 1:
    device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]  # List of available GPUs
else:  # If only one GPU is available, use cuda:0, else use CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

def RegisterModelClass(name):
    def regClass(cls):
        __private_key_value_models_map[name]=cls
    return regClass

model_keys_registered = __private_key_value_models_map.keys()
# Dictionary of models to be loaded in ModelConfig
def load_model_closure(model_name):
    models = __private_key_value_models_map
    return models[model_name]

# this is a hack till we add dynaconf or something?
if os.name == "nt":
    homepath = os.path.join('C:\\','Users',os.getlogin())
else:
    homepath = os.environ.get("HOME")

model_path_default = os.path.join( homepath , ".llm_vm", "models")
os.makedirs(model_path_default, exist_ok = True)

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
        return self.length
    def __getitem__(self, idx):
        return self.dataset[idx]

class BaseOnsiteLLM(ABC):
    def __init__(self,model_uri=None, tokenizer_kw_args={}, model_kw_args={}):
        if model_uri != None :
            self.model_uri= model_uri
        if model_uri is None and self.model_uri is None:
            raise ValueError('model_uri not found')
        self.model_name : str = self.model_uri.split('/')[-1] # our default for deriving model name
        self.model=self.model_loader(**model_kw_args)
        self.tokenizer=self.tokenizer_loader(**tokenizer_kw_args)

        # Move the model to the specified device(s)
        if isinstance(device, list):
            # If multiple GPUs are available, use DataParallel to parallelize the model
            self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(len(device))])
            self.model.to(device[0])  # Move model to the first GPU in the list
            print(f"`{self.model_uri}` loaded on {len(device)} GPUs.", file=sys.stderr)
        else:
            self.model.to(device)  # Move model to the selected device (single GPU or CPU)
            print(f"`{self.model_uri}` loaded on {device}.", file=sys.stderr)

    @property
    @abstractmethod
    def model_uri(self):
        pass

    @model_uri.setter
    def model_uri(self,val):
        self.model_uri=val # check if this is correct

    # model_name : str = self.model_uri.split('/')[-1]

    @abstractmethod
    def model_loader(self):
        pass

    @abstractmethod
    def tokenizer_loader(self):
        pass

    def load_finetune(self, model_filename):
        self.model.load_state_dict(torch.load(os.path.join(model_path_default,"finetuned_models", self.model_name, model_filename)))


    def generate(self,prompt,max_length=100, tokenizer_kwargs={}, generation_kwargs={}): # both tokenizer and model take kwargs :(
        """
        This function uses the class's llm and tokenizer to generate a response given a user's prompt

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response

        Example:
           >>> SmallLocalOpt.generate("How long does it take for an apple to grow?")
           I think it takes about a week for the apple to grow.
        """


        if isinstance(device, list):
            # If multiple GPUs are available, use first one
            inputs = self.tokenizer(prompt, return_tensors="pt", **tokenizer_kwargs).to(device[0])
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        generate_ids=self.model.generate(inputs.input_ids, max_length=max_length, **generation_kwargs)
        resp= self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        # need to drop the len(prompt) prefix with these sequences generally
        # because they include the prompt.
        return resp[len(prompt):]

    def finetune(self,data, optimizer, c_id, model_filename=None):
        def asynctune():
            old_model = optimizer.storage.get_model(c_id)
            if old_model is not None:
                self.model.load_state_dict(torch.load(old_model))
            untokenized_final_dataset = []
            for prompt,response in data:
                untokenized_final_dataset.append(prompt + response)
            tokenized_final_dataset = map(self.tokenizer,untokenized_final_dataset)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            optimizer.storage.set_training_in_progress(c_id, True)
            training_args = TrainingArguments(
                output_dir=os.path.join(model_path_default,"finetuned_models",),
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size = 1,
                num_train_epochs=5,
                weight_decay=0.01,
                report_to= "none",
            )
            test_set = FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset))

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=test_set,
                eval_dataset=test_set,
                data_collator=data_collator,
            )
            os.makedirs(os.path.join(model_path_default,"finetuned_models", self.model_name), exist_ok=True)
            if tokenized_final_dataset:
                trainer.train()
                eval_results = trainer.evaluate()
            optimizer.storage.set_training_in_progress(c_id, False)

            if os.name == "nt":
                timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            new_model = os.path.join(model_path_default,"finetuned_models",self.model_name, timestamp + '_' + self.model_name + ".pt" ) if model_filename is None else os.path.join(model_path_default,"finetuned_models",model_filename)
            open(new_model,"a")
            torch.save(self.model.state_dict(), new_model) # the model in memory is different now
            self.model_name = self.model_name + "_ft_"+  timestamp
            optimizer.storage.set_model(c_id, new_model)
            return math.exp(eval_results['eval_loss']) #perplexity is the metric we use for finetuning measurement
        return asynctune


    def lora_finetune(self, data, optimizer, c_id, model_filename=None):
        def async_lora():
            old_model = optimizer.storage.get_model(c_id)
            if old_model is not None:
                self.model.load_state_dict(torch.load(old_model))
            untokenized_final_dataset = []
            for prompt,response in data:
                untokenized_final_dataset.append(prompt + response)
            tokenized_final_dataset = map(self.tokenizer,untokenized_final_dataset)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            optimizer.storage.set_training_in_progress(c_id, True)
            training_args = TrainingArguments(
                output_dir=os.path.join(model_path_default,"finetuned_models",),
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size = 1,
                num_train_epochs=5,
                weight_decay=0.01,
                report_to= "none",
            )
            test_set = FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset))

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=test_set,
                eval_dataset=test_set,
                data_collator=data_collator,
                peft_config=peft_config
            )
            os.makedirs(os.path.join(model_path_default,"finetuned_models", self.model_name), exist_ok=True)
            if tokenized_final_dataset:
                trainer.train()
                eval_results = trainer.evaluate()
            optimizer.storage.set_training_in_progress(c_id, False)

            if os.name == "nt":
                timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            new_model = os.path.join(model_path_default,"finetuned_models",self.model_name, timestamp + '_' + self.model_name + ".pt" ) if model_filename is None else os.path.join(model_path_default,"finetuned_models",model_filename)
            open(new_model,"a")
            torch.save(self.model.state_dict(), new_model) # the model in memory is different now
            self.model_name = self.model_name + "_ft_"+  timestamp
            optimizer.storage.set_model(c_id, new_model)
            return math.exp(eval_results['eval_loss']) #perplexity is the metric we use for finetuning measurement
        return async_lora
    
    def quantize_model(self, bits=4):
        if self.model.is_quantizable():
            q4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
            q8_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            
            if bits == 4:
                q_model = AutoModelForCausalLM.from_pretrained(self.model_uri, quantization_config=q4_config)
            elif bits == 8:
                q_model = AutoModelForCausalLM.from_pretrained(self.model_uri, quantization_config=q8_config)
            else:
                raise ValueError("Only 4-bit and 8-bit quantization supported")
            return q_model
        else:
            raise NotImplementedError(f"{self.model} cannot be quantized")


    def qlora_finetune(self, data, optimizer, c_id, model_filename=None):
        def async_qlora():
            old_model = optimizer.storage.get_model(c_id)
            if old_model is not None:
                self.model.load_state_dict(torch.load(old_model))
            untokenized_final_dataset = []
            for prompt,response in data:
                untokenized_final_dataset.append(prompt + response)
            tokenized_final_dataset = map(self.tokenizer,untokenized_final_dataset)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            optimizer.storage.set_training_in_progress(c_id, True)
            training_args = TrainingArguments(
                output_dir=os.path.join(model_path_default,"finetuned_models",),
                evaluation_strategy="epoch",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=10,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                optim="paged_adamw_8bit"
            )
            test_set = FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset))
            
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["query_key_value"], 
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, config)
            trainer = Trainer(
                self.model,
                args=training_args,
                train_dataset=test_set,
                eval_dataset=test_set,
                data_collator=data_collator,
            )
            os.makedirs(os.path.join(model_path_default,"finetuned_models", self.model_name), exist_ok=True)
            if tokenized_final_dataset:
                trainer.train()
                eval_results = trainer.evaluate()
            optimizer.storage.set_training_in_progress(c_id, False)

            if os.name == "nt":
                timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            else:
                timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            new_model = os.path.join(model_path_default,"finetuned_models",self.model_name, timestamp + '_' + self.model_name + ".pt" ) if model_filename is None else os.path.join(model_path_default,"finetuned_models",model_filename)
            open(new_model,"a")
            torch.save(self.model.state_dict(), new_model) # the model in memory is different now
            self.model_name = self.model_name + "_ft_"+  timestamp
            optimizer.storage.set_model(c_id, new_model)
            return math.exp(eval_results['eval_loss']) #perplexity is the metric we use for finetuning measurement
        return async_qlora
    
    def finetune_immediately(self):
        self.finetune()()

    def lora_finetune_immediately(self):
        self.lora_finetune()()

    def qlora_finetune_immediately(self):
        self.qlora_finetune()()



"""
this factorization isn't necessarily the greatest, nor should it be viewed
as likely being more general, aside from covering hugging face transformers
"""
@RegisterModelClass("pythia")
class SmallLocalPythia(BaseOnsiteLLM):
    """
    This is a class for ElutherAI's Pythia-70m LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    # def __init__(self,**kwargs):
    #     # self.model_uri =
    #     super().__init__(kwargs) ## this line is required
    model_uri = "EleutherAI/pythia-70m-deduped"
    def model_loader(self):
        return GPTNeoXForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)


@RegisterModelClass("opt")
class SmallLocalOpt(BaseOnsiteLLM):

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
    model_uri="facebook/opt-350m"
    def model_loader(self):
        return OPTForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)

@RegisterModelClass("bloom")
class SmallLocalBloom(BaseOnsiteLLM):

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
    model_uri="bigscience/bloom-560m"

    def model_loader(self):
        return BloomForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)

@RegisterModelClass("neo")
class SmallLocalNeo(BaseOnsiteLLM):
    """
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="EleutherAI/gpt-neo-1.3B"

    def model_loader(self):
        return GPTNeoForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)


# Add support for "Open-Orca/LlongOrca-7B-16k"
@RegisterModelClass("smallorca")
class SmallLocalOpenOrca(BaseOnsiteLLM):
    """
    This is a class for Openlm-Research's open_orca-3b LLM
    
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
    
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="Open-Orca/LlongOrca-7B-16k"
    
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)


# Add support for "Open-Orca/LlongOrca-13B-16k"
@RegisterModelClass("orca")
class LocalOpenOrca2(BaseOnsiteLLM):
    """
    This is a class for Openlm-Research's open_orca-3b LLM
        
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
        
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="Open-Orca/LlongOrca-13B-16k"
        
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)


# Add support for "Open-Orca/Mistral-7B-OpenOrca"
@RegisterModelClass("mistral")
class SmallLocalOpenMistral(BaseOnsiteLLM):
    """
    This is a class for OpenOrca's Mistral-7b LLM
                
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
                
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="Open-Orca/Mistral-7B-OpenOrca"
                
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri) 


# Add support for "Open-Orca/OpenOrca-Platypus2-13B"
@RegisterModelClass("platypus")
class LocalOpenPlatypus(BaseOnsiteLLM):
    """
    This is a class for Open Orca's OpenOrca Platypus2-13b LLM
                
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
                
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="Open-Orca/OpenOrca-Platypus2-13B"
                
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)
    

@RegisterModelClass("llama")
class SmallLocalOpenLLama(BaseOnsiteLLM):
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
    model_uri="openlm-research/open_llama_3b_v2"

    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)


@RegisterModelClass("llama2")
class SmallLocalLLama(BaseOnsiteLLM):

    """
    This is a class for Meta's llama-7b LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="meta-llama/Llama-2-7b-hf"

    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)


@RegisterModelClass("codellama-7b")
class CodeLlama7b(BaseOnsiteLLM):
        
    """
    This is a class for Meta's code-llama-7b LLM
    
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
    
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="codellama/CodeLlama-7b-hf"
    
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return CodeLlamaTokenizer.from_pretrained(self.model_uri)
    
@RegisterModelClass("codellama-13b")
class CodeLlama13b(BaseOnsiteLLM):
        
    """
    This is a class for Meta's code-llama-13b LLM
    
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
    
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="codellama/CodeLlama-13b-hf"
    
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return CodeLlamaTokenizer.from_pretrained(self.model_uri)   
    
@RegisterModelClass("codellama-34b")
class CodeLlama34b(BaseOnsiteLLM):
        
    """
    This is a class for Meta's code-llama-34b LLM
    
    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model
    
    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """
    model_uri="codellama/CodeLlama-34b-hf"
    
    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return CodeLlamaTokenizer.from_pretrained(self.model_uri)   


@RegisterModelClass("flan")# our yummiest model based on similarity to food
class SmallLocalFlanT5(BaseOnsiteLLM):

    """
    This is a class for Google's flan-t5 LLM

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """

    model_uri="google/flan-t5-small"
    def model_loader(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)

@RegisterModelClass("bert")
class SmallLocalBERT(BaseOnsiteLLM):

    """
    This is a class for BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    The base model needs finetuning in almost all cases.

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        tokenizer (AutoTokenizer): Tokenizer from Transformer's library
        model (LLM): The large language model

    Methods:
        model_loader: Loads the LLM into memory
        tokenizer_loader: Loads the tokenizer into memory
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """

    model_uri = "bert-base-cased"
    def model_loader(self):
        return AutoModelForMaskedLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return AutoTokenizer.from_pretrained(self.model_uri)

@RegisterModelClass("gpt")
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
            >>> SmallLocalOpt.generate("How long does it take for an apple to grow?")
            It typically takes about 100-200 days...
        """

        ans = openai.completions.create(prompt= prompt, model="text-davinci-003", **kwargs)
        return ans.choices[0].text


    def finetune(self, dataset, optimizer, c_id, small_model_filename=None):
        old_model = optimizer.storage.get_model(c_id)
        training_file = create_jsonl_file(dataset)
        upload_response = openai.files.create(file=training_file, purpose="fine-tune", model="gpt-3.5-turbo-0613")
        training_file.close()
        fine_tuning_job = openai.fine_tunes.create(training_file= upload_response.id)

        print(f"Fine-tuning job created: {fine_tuning_job}", flush=True, file=sys.stderr)
        global job_id # global state isn't great, but thats interrupt handlers
        job_id = fine_tuning_job["id"]
        while True:
            fine_tuning_status = openai.fine_tunes.retrieve(id=job_id)
            status = fine_tuning_status["status"]
            print(f"Fine-tuning job status: {status}", file=sys.stderr)
            if status in ["succeeded", "completed", "failed"]:
                break
            time.sleep(30)
        job_id = None #
        new_model_id = fine_tuning_status.fine_tuned_model

        print("New_model_id: ", new_model_id, flush=True, file=sys.stderr)

        optimizer.storage.set_model(c_id, new_model_id)
        optimizer.storage.set_training_in_progress(c_id, False)
        if old_model is not None:
            openai.models.delete(old_model)


@RegisterModelClass("gpt4")
class GPT4:
    """
    This is a class for openAI's gpt-4 LLM

    Methods:
        generate: Generates a response from a given prompt through OpenAI's endpoint
    """

    def generate(self, prompt, max_length=100, **kwargs):
        """
        This function uses openAI's API to generate a response from the prompt using the GPT-4 model

        Parameters:
            prompt (str): Prompt to send to LLM
            max_length (int): Optional parameter limiting response length


        Returns:
            str: LLM Generated Response

        Example:
            >>> GPT4.generate("How long does it take for an apple to grow?")
            It typically takes about 100-200 days...
        """

        cur_prompt = [{'role': "system", 'content': prompt}]
        ans = openai.chat.completions.create(messages=cur_prompt,
        model="gpt-4",
        **kwargs)

        return ans.choices[0].message.content

    def finetune(self, dataset, optimizer, c_id, small_model_filename=None):
        print("fine tuning isn't supported by OpenAI on this model", file=sys.stderr)
        raise Exception("fine tuning isn't supported by OpenAI on this model")


@RegisterModelClass("chat_gpt")
class ChatGPT:
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
            >>> SmallLocalOpt.generate("How long does it take for an apple to grow?")
            It typically takes about 100-200 days...
        """
        cur_prompt = [{'role': "system", 'content' : prompt}]
        ans = openai.chat.completions.create(messages=cur_prompt,
        model="gpt-3.5-turbo-0301",
        **kwargs)
        return ans.choices[0].message.content

    def finetune(self, dataset, optimizer, c_id, small_model_filename=None):
        print("fine tuning isn't supported by OpenAI on this model", file=sys.stderr)
        raise Exception("fine tuning isn't supported by OpenAI on this model")
        # old_model = optimizer.storage.get_model(c_id)
        # training_file = create_jsonl_file(dataset)
        # upload_response = openai.File.create(file=training_file, purpose="fine-tune")
        # training_file.close()
        # fine_tuning_job = openai.FineTune.create(training_file= upload_response.id)

        # print(f"Fine-tuning job created: {fine_tuning_job}", flush=True)
        # global job_id # global state isn't great, but thats interrupt handlers
        # job_id = fine_tuning_job["id"]
        # while True:
        #     fine_tuning_status = openai.FineTune.retrieve(id=job_id)
        #     status = fine_tuning_status["status"]
        #     print(f"Fine-tuning job status: {status}")
        #     if status in ["succeeded", "completed", "failed"]:
        #         break
        #     time.sleep(30)
        # job_id = None #
        # new_model_id = fine_tuning_status.fine_tuned_model

        # print("New_model_id: ", new_model_id, flush=True)

        # optimizer.storage.set_model(c_id, new_model_id)
        # optimizer.storage.set_training_in_progress(c_id, False)
        # if old_model is not None:
        #     openai.Model.delete(old_model)


class BaseCtransformersLLM(BaseOnsiteLLM):
    """
    Base Class for running Ctransformers/GGML models

    Attributes:
        model_uri (str): Ctransformers uri for LLM
        model_kwargs (dict): Keyword arguments for loading the LLM
    
    Methods:
        model_loader: Loads specified model from Ctransformers
        generate: Generates a response from given prompt

    """

    def __init__(self, with_GPU=False, **model_kwargs):
        self.__model_uri = None
        self.__model_file = None
        if with_GPU:
            self.model = self.gpu_model_loader(**model_kwargs)
        else:
            self.model = self.model_loader(**model_kwargs)

    @property
    def model_file(self):
        return self.__model_file

    @model_file.setter
    def model_file(self,val):
        self.__model_file=val 

    @property
    def model_uri(self):
        return self.__model_uri

    @model_uri.setter
    def model_uri(self,val):
        self.__model_uri=val 
        
    def load_finetune(self, model_filename):
            raise Exception("Finetuning not supported for Ctransformers/GGML.")

    def _get_model_layers(self):
        pass

    def _get_model_size(self):
        pass    
    
    def model_loader(self, **kwargs):
        if 'model_file' in kwargs:
            del kwargs['model_file']

        if self.model_file is not None:
            return AutoModelForCausalLM.from_pretrained(self.__model_uri, model_file=self.__model_file, **kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(self.__model_uri, **kwargs)
        
    def gpu_model_loader(self, vram=0, **kwargs):
        if 'model_file' in kwargs:
            del kwargs['model_file']

        if vram > 0:
            model_size = self._get_model_size()
            model_layers = self._get_model_layers()

            size_per_layer = model_size // model_layers
            offload_layers = vram // size_per_layer
            if offload_layers > model_layers:
                offload_layers = model_layers

            if self.model_file is not None:
                return AutoModelForCausalLM.from_pretrained(self.__model_uri, model_file=self.__model_file, gpu_layers=offload_layers, **kwargs)
            else:
                return AutoModelForCausalLM.from_pretrained(self.__model_uri, gpu_layers=offload_layers, **kwargs)
        else: 
            raise ValueError("Expected VRAM to be greater than 0.")
        
    def generate(self, prompt, *generate_kwargs):  
        input_ids = self.model.tokenize(prompt)
        response = self.model.generate(input_ids, *generate_kwargs)
        return self.model.detokenize(response)
    
    def finetune(self, data, optimizer, c_id, model_filename=None):
        raise Exception("Finetuning not supported for Ctransformers/GGML.")


@RegisterModelClass("quantized-llama2-7b-base")
class Base_Llama2_7b_Q4(BaseCtransformersLLM):
    """
    Class for running quantized Llama 2 7b base model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-7B-GGML"
    model_file="llama-2-7b.ggmlv3.q4_K_M.bin"

@RegisterModelClass("quantized-llama2-13b-base")

class Base_Llama2_13b_Q4(BaseCtransformersLLM):
    """
    Class for running quantized Llama 2 13b base model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-13B-GGML"
    model_file="llama-2-13b.ggmlv3.q4_K_M.bin"


@RegisterModelClass("llama2-7b-chat-Q4")
class Chat_Llama2_7b_Q4(BaseCtransformersLLM):
    """
    Class for running Llama2-7b-Chat model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-7B-Chat-GGML"
    model_file="llama-2-7b-chat.Q4_K_M.gguf"

@RegisterModelClass("llama2-7b-chat-Q6")
class Chat_Llama2_7b_Q6(BaseCtransformersLLM):
    """
    Class for running Llama2-7b-Chat model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-7B-Chat-GGML"
    model_file="llama-2-7b-chat.Q6_K.gguf"
 

@RegisterModelClass("llama2-13b-chat-Q4")
class Chat_Llama2_13b_Q4(BaseCtransformersLLM):
    """
    Class for running Llama2-13b-Chat model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-13B-Chat-GGML"
    model_file="llama-2-13b-chat.Q4_K_M.gguf"


@RegisterModelClass("llama2-13b-chat-Q6")
class Chat_Llama2_13b_Q6(BaseCtransformersLLM):
    """
    Class for running Llama2-13b-Chat model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-13B-Chat-GGML"
    model_file="llama-2-13b-chat.Q6_K.gguf"


@RegisterModelClass("llama2-7b-32k-Q4")
class Chat_Llama2_13b_Q6(BaseCtransformersLLM):
    """
    Class for running Llama2-7b-32k-Instruct model instance

    Properties:
        model_uri: Ctransformers uri for LLM
        model_file: gguf or bin file for repos with multiple files
    """

    model_uri="TheBloke/Llama-2-7B-32K-Instruct-GGML"
    model_file="llama-2-7b-32k-instruct.ggmlv3.q4_1.bin"
