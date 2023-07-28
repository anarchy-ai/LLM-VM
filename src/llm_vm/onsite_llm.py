import abc
from abc import ABC,abstractmethod
import openai
import math
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    OPTForCausalLM,
    BloomForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer)
import time
from datetime import datetime
import tempfile
import json
import os
import torch


__private_key_value_models_map =  {}
# []   {
#         "opt": Small_Local_OPT,
#         "bloom": Small_Local_Bloom,
#         "neo": Small_Local_Neo,
#         "llama": Small_Local_LLama,
#         "pythia": Small_Local_Pythia,
#         "gpt": GPT3,
#         "chat_gpt": Chat_GPT,
#         "flan" : Small_Local_Flan_T5,
#         "pythia" : Small_Local_Pythia,
#         }

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

class Base_Onsite_LLM(ABC):
    def __init__(self,model_uri=None,tokenizer_kw_args={},model_kw_args={}):
        if model_uri != None :
            self.model_uri= model_uri
        if model_uri is None and self.model_uri is None:
            raise ValueError('A very specific bad thing happened.')
        self.model_name : str = self.model_uri.split('/')[-1] # our default for deriving model name
        self.model=self.model_loader(**model_kw_args)
        self.tokenizer=self.tokenizer_loader(**tokenizer_kw_args)

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
        # because they include the prompt.
        return resp[len(prompt):]

    def finetune(self,data, optimizer, c_id):
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
                num_train_epochs=1,
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
            new_model = os.path.join(model_path_default,"finetuned_models",self.model_name, timestamp + '_' + self.model_name + ".pt" )
            open(new_model,"a")
            torch.save(self.model.state_dict(), new_model) # the model in memory is different now
            self.model_name = self.model_name + "_ft_"+  timestamp
            optimizer.storage.set_model(c_id, new_model)
            return math.exp(eval_results['eval_loss']) #perplexity is the metric we use for finetuning measurement
        return asynctune


    def finetune_immediately(self):
        finetune()()


class TokenStreamerDisguisedAsStoppingCriterion:
    def __init__(self, token_streamer):
        self.token_streamer = token_streamer

    def __call__(self, input_ids, scores, **kwargs):
        self.token_streamer(input_ids, scores, **kwargs)
        return False


class TokenStreamer:
    def __call__(self, input_ids, scores, **kwargs):
        raise NotImplementedError


class HFTransformers:
    def __init__(self, model_uri, **kwargs):
        self.model_identifier = model_uri
        self.model_args = kwargs

        from transformers import AutoModelForCausalLM
        print(f"Creating {self.model_identifier} instance using AutoModelForCausalLM transformers module", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier, **self.model_args)
        
        print(f"{self.model_identifier} model is ready for use on {self.model.device}", flush=True)

        self.max_batch_size = kwargs.get("batch_size", 8)

    @property
    def eos_token_id(self):
        return self.model.config.eos_token_id

    def score(self, input_ids, attention_mask, **model_kwargs):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        # prepare model inputs
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs, eos_token_id=self.eos_token_id)
        print(model_input)
        model_inputs["attention_mask"] = attention_mask

        token_scores = []
        
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -len(input_ids), :]
        next_token_logits = torch.log_softmax(next_token_logits, dim=-1)
        token_scores = next_token_logits.gather(-1, input_ids)

        return token_scores
    
    def generate(self, input_ids, attention_mask, temperature, max_new_tokens, bias_tensor, streamer):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        kwargs = {
            "input_ids": input_ids.to(self.model.device),
            "do_sample": temperature > 0.0,
            "attention_mask": attention_mask.to(self.model.device),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "logits_processor": self.logits_processors(bias_tensor),
            "output_scores": True,
            "return_dict_in_generate": True
        }

        result = self.model.generate(**kwargs, stopping_criteria=[TokenStreamerDisguisedAsStoppingCriterion(streamer)], 
                                     eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id)

        return (result.sequences, result.scores)
    
    def make_bias_tensor(self, logit_biases, vocab_size):
        bias_tensors = [torch.zeros(vocab_size) for _ in logit_biases]
        for i, bias in enumerate(logit_biases):
            if len(bias) > 0:
                indices = torch.tensor(list(bias.keys()), dtype=torch.int64)
                values = torch.tensor(list(bias.values()), dtype=torch.float32)
                bias_tensors[i][indices] = values
        return torch.stack(bias_tensors)
    
    def logits_processors(self, logit_biases):
        bias_tensors = None
        make_bias_tensor = self.make_bias_tensor
        
        if len(logit_biases) == 0:
            return []

        class BatchLogitsProcessor:
            def __call__(self, input_ids, scores):
                nonlocal bias_tensors

                if bias_tensors is None:
                    bias_tensors = torch.tensor(make_bias_tensor(logit_biases, scores.shape[-1])).to(scores.device)

                return scores + bias_tensors

        return [BatchLogitsProcessor()]
    

"""
this factorization isn't necessarily the greatest, nor should it be viewed
as likely being more general, aside from covering hugging face transformers
"""
@RegisterModelClass("pythia")
class Small_Local_Pythia(Base_Onsite_LLM):
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
class Small_Local_OPT(Base_Onsite_LLM):

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
class Small_Local_Bloom(Base_Onsite_LLM):

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
class Small_Local_Neo(Base_Onsite_LLM):

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
        return GPT2Tokenizer.from_pretrained(self.model_uri)

@RegisterModelClass("llama")
class Small_Local_LLama(Base_Onsite_LLM):

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

@RegisterModelClass("flan")# our yummiest model based on similarity to food
class Small_Local_Flan_T5(Base_Onsite_LLM):

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
class Small_Local_BERT(Base_Onsite_LLM):

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
        return BertTokenizer.from_pretrained(self.model_uri)
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

@RegisterModelClass("chat_gpt")
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
        print("fine tuning isn't supported by OpenAI on this model")
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



if __name__ == "__main__":
    model = HFTransformers("facebook/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    input_text = "The person performing the heart surgery is a"
    input_ids = tokenizer(input_text, return_tensors="pt")
    print(input_ids)
    model_input = input_ids["input_ids"]
    print(model_input)
    class DebugStreamer:
        def __call__(self, tokens, scores):
            print(tokenizer.decode(tokens[-1]))
            return False
    output = model.generate(model_input, torch.ones_like(model_input), 1.0, 50, [], DebugStreamer())
    # print("Response: ", output)