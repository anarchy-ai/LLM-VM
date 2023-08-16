import abc
from abc import ABC,abstractmethod
import openai
import math
from transformers import (
    AutoModelForCausalLM,
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
    Trainer,
    LogitsProcessorList,
    LogitsProcessor
    )
import time
from datetime import datetime
import tempfile
import json
import os
import torch
import regex as re
from itertools import chain, combinations
from lark import Lark
from lark.indenter import PythonIndenter
import asyncio


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


class TokenConstraint(ABC):
    def __init__(self, model_uri):
        self.model_uri = model_uri
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_uri, add_prefix_space=True)
        self.parser = None

    @abstractmethod
    def construct_filter_set(self, expression):
       pass

    
class RegexTokenConstraint(TokenConstraint):
    def construct_filter_set(self, expression):
        vocab = self.tokenizer.vocab
        vocab_map = {v: k for k, v in vocab.items()}
        space_repr = self.tokenizer.tokenize(" ")[0]
        nl_repr = self.tokenizer.tokenize("\n")[0]
        expression = expression.replace(" ", space_repr)
        expression = expression.replace("\n", nl_repr)
        valid_tokens = []
        pattern = re.compile(expression, re.UNICODE)
        for subtoken in vocab_map.values():
            if pattern.match(subtoken) is not None:
                valid_tokens.append(subtoken)
        return valid_tokens


class PythonTokenConstraint(TokenConstraint): 

        async def _async_generator(self, v_map, pattern):
            for i in v_map.values():
                if pattern.match(i) is not None:
                    yield i

        async def _async_generator_final(self, v_set, pattern):
            for v in v_set:
                if pattern.match(v) is not None:
                    yield v

        async def construct_filter_set(self, expressions):
            vocab = self.tokenizer.vocab
            vocab_map = {v: k for k, v in vocab.items()}
            valid_tokens = []

            for expression in expressions:
                space_repr = self.tokenizer.tokenize(" ")[0]
                nl_repr = self.tokenizer.tokenize("\n")[0]
                expression = expression.replace(" ", space_repr)
                expression = expression.replace("\n", nl_repr)

                try:  
                    pattern = re.compile(expression, re.UNICODE)
                    async for token in self._async_generator(vocab_map, pattern):
                        valid_tokens.append(token)
                except Exception as e: 
                    print(e) 
      
            return set(valid_tokens)
        
        def parse_grammar(self):
            self.parser = Lark.open_from_package('lark', 'python.lark', ['grammars'], parser='lalr', regex=True, lexer='contextual', postlex=PythonIndenter(), start='file_input')
            terminals = self.parser.terminals
            t_map = {}
            for t in terminals:
                t_map[t.name] = t.pattern.value
            return t_map

        def prefix_state(self, prefix_str):
            valid_next = []
            try:      
                self.parser.parse(prefix_str)
            except:
                interactive_tree = self.parser.parse_interactive(prefix_str)
                interactive_tree.exhaust_lexer()
                valid_next = list(interactive_tree.accepts())
            return valid_next

        async def construct_final_filter_set(self, prefix_str, token_map, crude_filter_set):
            next_lex = self.prefix_state(prefix_str)
            final_tokens_regex = []
            final_tokens = []
            for lex in next_lex:
                for k, v in token_map.items():
                    if lex == k:
                        final_tokens_regex.append(v)
            
            for expression in final_tokens_regex:
                space_repr = self.tokenizer.tokenize(" ")[0]
                nl_repr = self.tokenizer.tokenize("\n")[0]
                expression = expression.replace(" ", space_repr)
                expression = expression.replace("\n", nl_repr)

                try:  
                    pattern = re.compile(expression, re.UNICODE)
                    async for token in self._async_generator_final(crude_filter_set, pattern):
                        final_tokens.append(token)
                except Exception as e: 
                    print(e)
            return set(final_tokens)
            

class RegexLogitsProcessor(LogitsProcessor):
   
        def __init__(self, sequence_bias):
            self.sequence_bias = sequence_bias
            self._validate_arguments()
            self.sequences_length_greater_than_1 = []
            self.length_1_bias = None
            self.length_greater_than_1_bias = None
            self.prepared_bias_variables = False

        def __call__(self, input_ids, scores):
            if not self.prepared_bias_variables:
                self._prepare_bias_variables(scores)

            bias = torch.zeros_like(scores)

            bias += self.length_1_bias

            matching_mask = torch.zeros_like(scores, dtype=torch.bool)
            for sequence_ids in self.sequences_length_greater_than_1:
                if len(sequence_ids) > input_ids.shape[1]: 
                    continue
                prefix_length = len(sequence_ids) - 1
                last_token = sequence_ids[-1]
                matching_rows = torch.eq(
                    input_ids[:, -prefix_length:],
                    torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
                ).prod(dim=1)
                matching_mask[:, last_token] |= matching_rows.bool()
            bias += torch.where(
                matching_mask,
                self.length_greater_than_1_bias,
                torch.tensor(0.0, device=self.length_greater_than_1_bias.device),
            )

            scores = scores + bias
            return scores

        def _prepare_bias_variables(self, scores):
            vocabulary_size = scores.shape[-1]
            sequence_bias = self.sequence_bias
            tokens_with_bias = []
            self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float)
            self.length_greater_than_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float)
            for sequence_ids, bias in sequence_bias.items():
                if len(sequence_ids) == 1:
                    self.length_1_bias[sequence_ids[-1]] = bias
                else:
                    self.sequences_length_greater_than_1.append(sequence_ids)
                    if self.length_greater_than_1_bias[sequence_ids[-1]] != 0.0:
                        raise ValueError(
                            "Setting a bias on sequences that share a common token termination is not supported."
                        )
                    self.length_greater_than_1_bias[sequence_ids[-1]] = bias
                tokens_with_bias.append(sequence_ids[-1])

            self.prepared_bias_variables = True

        def _validate_arguments(self):
            sequence_bias = self.sequence_bias
            if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
                raise ValueError(f"sequence_bias has to be a non-empty dictionary, but is {sequence_bias}.")


class HF_LLM(ABC):
    def __init__(self, model_uri, tokenizer, **kwargs):
        self.model_identifier = model_uri
        self.model_args = kwargs
        self.tokenizer = tokenizer

        print(f"Creating {self.model_identifier} instance using AutoModelForCausalLM transformers module", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier, **self.model_args)
        print(f"{self.model_identifier} model is ready for use on {self.model.device}", flush=True)
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab = self.tokenizer.vocab

    @property
    def eos_token_id(self):
        return self.model.config.eos_token_id
    
    @abstractmethod
    def generate(self, constraint_type=None, **kwargs):
       pass


class HFTransformers(HF_LLM):
    
    def generate(self, constraint_type=None, **kwargs):
        if constraint_type is not None:
            raise TypeError('Expected constraint_type to be None')
        result = self.model.generate(**kwargs, eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id)
        return (result.sequences, result.scores)
    

class HFTransformersWithConstraints(HF_LLM):
    
    def generate(self, constraint_type=None, **kwargs):
        if type(constraint_type) is not str:
            raise TypeError('Expected constraint_type to be a string')
        assert torch.is_tensor(kwargs["input_ids"]), "Input ids must be a torch tensor"
        
        if constraint_type == "regex":
            if 'r_bias' not in kwargs:
                r_bias = 10.0
            else:
                r_bias= float(kwargs['r_bias'])
            r_constraints = RegexTokenConstraint(self.model_identifier) 
            valid_tokens = r_constraints.construct_filter_set(kwargs['regex'])
            del kwargs['regex']
            seq_bias = {}
            for t in valid_tokens:
                t_tuple = tuple(self.tokenizer.encode(t, add_special_tokens=False))
                seq_bias[t_tuple] = r_bias
            logits_processor = LogitsProcessorList()
            logits_processor.append(RegexLogitsProcessor(seq_bias))
        elif constraint_type == "grammar":
            if 'language' not in kwargs: 
                raise Exception("language must be specified in kwargs")
            if kwargs['language'] == 'python':
                python_constraint = PythonTokenConstraint(self.model_identifier)
                terminals_map = python_constraint.parse_grammar()
                valid_tokens = []
                tokens_map = {}
                for k, v in terminals_map.items():
                    v = str(v)
                    matching_tokens = python_constraint.construct_filter_set(v)
                    tokens_map[k] = matching_tokens
                    valid_tokens += matching_tokens
                logits_processor = LogitsProcessorList()
                valid_tokens = set(valid_tokens)
                del kwargs['language']
            else: 
                raise Exception(f"{kwargs['language']} is not supported")
        else:
            raise Exception(f"{constraint_type} not supported")
        
        kwargs["logits_processor"] = logits_processor
        kwargs["do_sample"] =  kwargs["temperature"] > 0
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True

        result = self.model.generate(**kwargs, eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id)
        return (result.sequences, result.scores)

if __name__ == "__main__":

    # python_parser = Lark.open_from_package('lark', 'python.lark', ['grammars'], parser='lalr', lexer='contextual', postlex=PythonIndenter(), start='file_input')
    # terminals = python_parser.terminals
    # t_map = {}
    # for t in terminals:
    #     t_map[t.name] = t.pattern

    # interactive_tree = python_parser.parse_interactive("name=1\n")
    # interactive_tree.exhaust_lexer()
    # print(interactive_tree.accepts())
    # # def handle_errors(e):
    # #     print(e)
    # #     return True
    # parse_tree = python_parser.parse("name=1\nres=3 + name\n")
    async def main():
        python_constraint = PythonTokenConstraint("facebook/opt-350m")
        terminals_map = python_constraint.parse_grammar()
        valid_tokens = []
        tokens_map = {}
        
        for k, v in terminals_map.items():
            valid_tokens.append(v)
        print(valid_tokens)
        res = await python_constraint.construct_filter_set(valid_tokens)
        print(res)



    asyncio.run(main())
    
    # main()
    #     tokens_map[k] = matching_tokens
    #     valid_tokens += matching_tokens
    # res = python_constraint.construct_filter_set('(?:(?:\r?\n[\t ]*|#[^\n]*))+')
    # res = python_constraint.construct_final_filter_set("name=1\n", tokens_map)
    # print(res)



    
    # re_str = "doctor|specialist"
    # # res = interface.construct_crude_filter_set(re_str)
    # # print(res)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", add_prefix_space=True)
    # input_text = "The one performing the heart surgery is a"
    # token_ids = tokenizer(input_text, return_tensors="pt")
    # kwargs = {
    #     'regex': re_str,
    #     'input_ids': token_ids['input_ids'],
    #     'attention_mask': token_ids['attention_mask'],
    #     'temperature': 1.0,
    #     'max_new_tokens': 10
    # }
    # print("Input: ", token_ids['input_ids'] )
    # model = HFTransformersWithConstraints("facebook/opt-350m", tokenizer)
    # res = model.generate(constraint_type="regex", **kwargs)
    # res_text = tokenizer.batch_decode(res[0], skip_special_tokens=True)
    # print("Response: ", res_text)