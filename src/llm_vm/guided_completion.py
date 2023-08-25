import outlines.models as models 
import outlines.text.generate as generate
import torch
from lark import Lark, Transformer, v_args
from lark.indenter import PythonIndenter
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor)
import re
from abc import ABC,abstractmethod

model = models.transformers("gpt2-medium")

#this class is called when optimize.complete is called with the regex parameter
class RegexCompletion:
    def complete(prompt,regex):
        guided = generate.regex(model,regex)(prompt)
        return guided

#this class is called when optimize.complete is called with the choices parameter
class ChoicesCompletion:
    def complete(prompt,choices):
        guided = generate.choice(model,choices)(prompt)
        return guided
    
#this class is called with optimize.complete is caled with the type parameter
class TypeCompletion:
    def complete(prompt,type):
        if type != "float" and type != "integer":
            raise Exception("type must be float or integer")
        guided = getattr(generate,type)(model)(prompt)
        return guided
    
class GrammarCompletion:
    def __init__(self, model_uri, tokenizer):
        self.model_identifier = model_uri
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier)
        print(f"{self.model_identifier} model is ready for use on {self.model.device}", flush=True)
        self.supported_grammar = ["python", "json"]
        self.terminals_tokens_map = {}

    @property
    def eos_token_id(self):
        return self.model.config.eos_token_id
    
    def complete(self, prompt, grammar_type='python'):
        if grammar_type.lower() not in self.supported_grammar:
            raise ValueError(f'{grammar_type} is not supported. The only valid grammar types are {self.supported_grammar}')
        
        token_ids = self.tokenizer(prompt, return_tensors='pt')
        input_len = len(token_ids['input_ids'][-1])

        if grammar_type.lower() == 'python':
            python_constraint = PythonConstraint(self.model_identifier, self.tokenizer)
            if(len(self.terminals_tokens_map) == 0):
                terminal_regex_map = python_constraint.parse_grammar()
                for k, v in terminal_regex_map.items():
                    matching_tokens = python_constraint.construct_filter_set(v)
                    self.terminals_tokens_map[k] = matching_tokens
                self.terminals_tokens_map['$END'] = [(self.tokenizer.decode(self.eos_token_id), self.eos_token_id)]

            print('Created token map')

            logits_processor = LogitsProcessorList()
            logits_processor.append(GrammarLogitsProcessor(python_constraint, self.terminals_tokens_map))

        model_kwargs = {
        'input_ids': token_ids['input_ids'],
        'attention_mask': token_ids['attention_mask'],
        'temperature': 1.0,
        'max_new_tokens': 20,
        }
        model_kwargs["do_sample"] =  model_kwargs["temperature"] > 0
        model_kwargs["output_scores"] = True
        model_kwargs["return_dict_in_generate"] = True
        model_kwargs["logits_processor"] = logits_processor
        res = self.model.generate(**model_kwargs, eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id)
        res_text = self.tokenizer.batch_decode(res.sequences[:, input_len:], skip_special_tokens=True)[0]
        return res_text.strip()
    

class GrammarConstraint(ABC):
    def __init__(self, model_uri, tokenizer):
        self.model_uri = model_uri
        self.tokenizer = tokenizer
        self.parser = None

    @abstractmethod
    def construct_filter_set(self, expression):
        pass

class PythonConstraint(GrammarConstraint): 

    def construct_filter_set(self, expression):
        vocab = self.tokenizer.vocab
        valid_tokens = []
        specials = ['*', '+', ')']
        if expression[0] in specials:
            expression = f'\{expression}'
        elif len(expression) == 1 and (expression == '[' or  expression == '('):
            expression = f'\{expression}'
        
        try:
            pattern = re.compile(expression, re.UNICODE)
            for token, id in vocab.items():
                if pattern.match(token) is not None:
                   valid_tokens.append((token, id))
        except Exception as e: 
            print(e, expression) 
    
        return set(valid_tokens)
    

    def parse_grammar(self):
        # Create the Python parser with Lark, using the LALR algorithm
        self.parser = Lark.open_from_package('lark', 'python.lark', ['grammars'], parser='lalr', regex=True, lexer='contextual', postlex=PythonIndenter(), start='file_input')
        terminals = self.parser.terminals
        t_map = {}
        for t in terminals:
            t_map[t.name] = t.pattern.value
        return t_map

    def prefix_state(self, prefix_str):
        valid_next = []
        try:      
            interactive_tree = self.parser.parse_interactive(prefix_str)
            interactive_tree.exhaust_lexer()
            valid_next = list(interactive_tree.accepts())
        except Exception as e:
            print(e)
        return valid_next

    def construct_final_filter_set(self, prefix_ids, terminals_map):
        prefix_str = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)[0]
        print(prefix_str)
        next_lex = self.prefix_state(prefix_str)
        valid_next_ids = []
        for lex in next_lex:
            token_set = terminals_map[lex]
            for t in token_set:
                valid_next_ids.append(t[-1])
        
        return set(valid_next_ids)

