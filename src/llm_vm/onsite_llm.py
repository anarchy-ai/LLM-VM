import json
import math
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime

import openai
import torch
from ctransformers import AutoModelForCausalLM
from peft import LoraConfig
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
    DataCollatorForLanguageModeling,
    TrainingArguments,
    BitsAndBytesConfig)
from trl import SFTTrainer

__private_key_value_models_map =  {}
# []   {
#         "opt": SmallLocalOpt,
#         "bloom": SmallLocalBloom,
#         "neo": SmallLocalNeo,
#         "llama": SmallLocalOpenLLama,
#         "llama2": SmallLocalLLama,
#         "gpt": GPT3,
#         "chat_gpt": ChatGPT,
#         "flan" : SmallLocalFlanT5,
#         "pythia" : SmallLocalPythia,
#         }

# default lora config
lora_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM",
)

# Combine all the configs
combination_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Load weights in 4-bit format for memory efficiency.
    bnb_4bit_quant_type="nf4",             # Specify the use of NF4 quantization for compression.
    bnb_4bit_use_double_quant=True,        # Apply double quantization for additional compression.
    bnb_4bit_compute_dtype=torch.bfloat16  # Set compute data type to bfloat16 for faster training.
)

# Use double quantization if memory is a concern.
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Load weights in 4-bit format.
    bnb_4bit_use_double_quant=True      # Apply double quantization for additional compression.
)

# Use a 16-bit compute dtype for faster fine-tuning.
bfloat16_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Load weights in 4-bit format.
    bnb_4bit_compute_dtype=torch.bfloat16  # Set compute data type to bfloat16 for faster training.
)

# Use NF4 quantization for higher precision.
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Load weights in 4-bit format.
    bnb_4bit_quant_type="nf4"           # Use NF4 quantization.
)

def formatting_func(example):
    """
    This function formats the example for the SFTTrainer.

    Parameters:
        example (dict): Dictionary containing the question and answer

    Returns:
        str: Formatted string containing the question and answer
    """
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


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
    def __init__(self,model_uri=None,tokenizer_kw_args={},model_kw_args={}):
        if model_uri != None :
            self.model_uri= model_uri
        if model_uri is None and self.model_uri is None:
            raise ValueError('A very specific bad thing happened.')
        self.model_name : str = self.model_uri.split('/')[-1]  # our default for deriving model name
        self.model=self.model_loader(**model_kw_args)
        self.tokenizer=self.tokenizer_loader(**tokenizer_kw_args)

        # set `quantization_config` to one of the following:
        # `combination` for all the configs (default)
        # `double_quant` for memory
        # `bfloat16` for speed
        # `nf4` for precision
        quantization_config = model_kw_args.get("quantization_config")
        if quantization_config == "combination" or quantization_config is None:
            self.model.quantization_config = combination_bnb_config
        elif quantization_config == "double_quant":
            self.model.quantization_config = double_quant_config
        elif quantization_config == "bfloat16":
            self.model.quantization_config = bfloat16_config
        elif quantization_config == "nf4":
            self.model.quantization_config = nf4_config

    @property
    @abstractmethod
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

    def load_finetune(self, model_filename):
        self.model.load_state_dict(torch.load(os.path.join(model_path_default,"finetuned_models", self.model_name, model_filename)))

    def generate(self,prompt,max_length=100,**kwargs): # both tokenizer and model take kwargs :(
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
        inputs=self.tokenizer(prompt,return_tensors="pt")
        generate_ids=self.model.generate(inputs.input_ids,max_length=max_length)
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
            # TODO(us): `SFTTrainer.packing` parameter not supported when `trl.DataCollatorForCompletionOnlyLM` used. Which one is more effective use that one.
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
                gradient_checkpointing=True  # Gradient checkpointing is a technique used to reduce memory consumption
                # during the training of deep neural networks, especially in situations where memory usage is a limiting
                # factor. Gradient checkpointing selectively re-computes intermediate activations during the backward
                # pass instead of storing them all, thus performing some extra computation to reduce memory usage.
            )
            test_set = FinetuningDataset(tokenized_final_dataset,len(untokenized_final_dataset))

            #  SFTTrainer is same as Trainer but accepts a peft config, so it can run lora fine-tuning
            #  (https://discuss.huggingface.co/t/when-to-use-sfttrainer/40998/4)
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=test_set,
                eval_dataset=test_set,
                data_collator=data_collator,
                peft_config=lora_config,
                packing=True,
                formatting_func=formatting_func,
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


    def finetune_immediately(self):
        finetune()()

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
    
@RegisterModelClass("quantized-llama")
class Quantized_Llama(Base_Onsite_LLM):
    """
    Class for running quantized Llama instances that use GGML

    Attributes:
        model_uri (str): Hugging Face Endpoint for LLM
        model (LLM): The large language model. CTransformers includes its own tokenizer in the model

    Methods:
        __init__: Takes the same parameters as the base class, except it looks to see if there is a specific quantization you want to use
        Quantized weights can be found (here)[https://huggingface.co/TheBloke/LLaMa-7B-GGML#provided-files]
        model_loader: Loads the LLM into memory
        tokenizer_loader: Does nothing. CTransformers includes its own tokenizer so this is unnecessary
        generate: Generates a response from a given prompt with the loaded LLM and tokenizer
    """

    model_uri="TheBloke/LLaMa-7B-GGML"

    def __init__(self,model_uri=None,tokenizer_kw_args={},model_kw_args={}):
        # Pop because we don't want to pass model_file onto the super class.
        self.model_file = model_kw_args.pop('model_file', None)
        if self.model_file is None:
        #     # Set default to smallest quantization available
            self.model_file = "llama-7b.ggmlv3.q2_K.bin"
        super().__init__(model_uri, tokenizer_kw_args, model_kw_args)


    def model_loader(self):
        # This file specifically is the smallest model.
        return AutoModelForCausalLM.from_pretrained(self.model_uri, model_file=self.model_file)

    # CTransformers loads its tokenizer in the model, so this function is unnecessary
    def tokenizer_loader(self):
        return

    def generate(self, prompt, max_length=100, **kwargs):
        inputs = self.model.tokenize(prompt)
        generated_tokens = self.model.generate(inputs)
        resp = (self.model.detokenize(generated_tokens))
        return resp

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
    model_uri="meta-llama/Llama-2-7b"

    def model_loader(self):
        return LlamaForCausalLM.from_pretrained(self.model_uri)
    def tokenizer_loader(self):
        return LlamaTokenizer.from_pretrained(self.model_uri)

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

        ans = openai.Completion.create(prompt= prompt, model="text-davinci-003", **kwargs)
        return ans['choices'][0]['text']


    def finetune(self, dataset, optimizer, c_id, small_model_filename=None):
        old_model = optimizer.storage.get_model(c_id)
        training_file = create_jsonl_file(dataset)
        upload_response = openai.File.create(file=training_file, purpose="fine-tune", model="gpt-3.5-turbo-0613")
        training_file.close()
        fine_tuning_job = openai.FineTune.create(training_file= upload_response.id)

        print(f"Fine-tuning job created: {fine_tuning_job}", flush=True, file=sys.stderr)
        global job_id # global state isn't great, but thats interrupt handlers
        job_id = fine_tuning_job["id"]
        while True:
            fine_tuning_status = openai.FineTune.retrieve(id=job_id)
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
            openai.Model.delete(old_model)

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
        ans = openai.ChatCompletion.create(
            messages=cur_prompt,
            model="gpt-3.5-turbo-0301",
            **kwargs)
        return ans['choices'][0]['message']['content']

    def finetune(self, dataset, optimizer, c_id):
        print("fine tuning isn't supported by OpenAI on this model", file=sys.stderr)
        exit()
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
