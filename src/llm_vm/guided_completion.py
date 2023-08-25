import outlines.models as models 
import outlines.text.generate as generate
import torch
from lark import Lark, Transformer, v_args
from lark.indenter import PythonIndenter
from transformers import (AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor)
import re
from abc import ABC,abstractmethod

model = models.transformers("gpt2")

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
        # Initialize the GrammarCompletion class with a model URI and tokenizer
        self.model_identifier = model_uri
        self.tokenizer = tokenizer

        # Load model from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier)
        print(f"{self.model_identifier} model is ready for use on {self.model.device}", flush=True)
        
        # Define supported grammar types 
        self.supported_grammar = ["python", "json"]

        # Initialize dict to store terminal symbols and their corresponding token mappings
        self.terminals_tokens_map = {}

    @property
    def eos_token_id(self):
        return self.model.config.eos_token_id
    
    def complete(self, prompt, grammar_type='python', **model_kwargs):
        # Check if the specified grammar type is supported
        if grammar_type.lower() not in self.supported_grammar:
            raise ValueError(f'{grammar_type} is not supported. The only valid grammar types are {self.supported_grammar}')

        # Encode the input prompt
        token_ids = self.tokenizer(prompt, return_tensors='pt')
        input_len = len(token_ids['input_ids'][-1])

        if grammar_type.lower() == 'python':
            # For Python grammar, we create a PythonConstraint instance and parse the grammar
            python_constraint = PythonConstraint(self.model_identifier, self.tokenizer)

            # Construct terminal tokens mapping if it's not yet initialized
            if(len(self.terminals_tokens_map) == 0):
                terminal_regex_map = python_constraint.parse_grammar()
                for k, v in terminal_regex_map.items():
                    matching_tokens = python_constraint.construct_filter_set(v)
                    self.terminals_tokens_map[k] = matching_tokens
                
                # Add end-of-sequence token mapping
                self.terminals_tokens_map['$END'] = [(self.tokenizer.decode(self.eos_token_id), self.eos_token_id)]

            # Initialize the LogitsProcessorList for processing sequence logits and append GrammarLogitsProcessor instance
            logits_processor = LogitsProcessorList()
            logits_processor.append(GrammarLogitsProcessor(python_constraint, self.terminals_tokens_map))
        elif grammar_type.lower() == 'json':
            # For JSON grammar, we create a JSONConstraint instance and parse the grammar
            json_constraint = JSONConstraint(self.model_identifier, self.tokenizer)

            # Construct terminal tokens mapping if it's not yet initialized
            if(len(self.terminals_tokens_map) == 0):
                terminal_regex_map = json_constraint.parse_grammar()
                for k, v in terminal_regex_map.items():
                    matching_tokens = json_constraint.construct_filter_set(v)
                    self.terminals_tokens_map[k] = matching_tokens
                
                # Add end-of-sequence token mapping
                self.terminals_tokens_map['$END'] = [(self.tokenizer.decode(self.eos_token_id), self.eos_token_id)]

            # Initialize the LogitsProcessorList for processing sequence logits and append GrammarLogitsProcessor instance
            logits_processor = LogitsProcessorList()
            logits_processor.append(GrammarLogitsProcessor(json_constraint, self.terminals_tokens_map))

        model_kwargs["do_sample"] =  model_kwargs["temperature"] > 0
        model_kwargs["output_scores"] = True
        model_kwargs["return_dict_in_generate"] = True
        model_kwargs["logits_processor"] = logits_processor

        # Generate completion using the model
        res = self.model.generate(**model_kwargs, input_ids=token_ids['input_ids'], attention_mask=token_ids['attention_mask'], eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id)

        # Decode completion and return the generated text, including input tokens
        res_text = self.tokenizer.batch_decode(res.sequences, skip_special_tokens=True)[0]
        return res_text.strip()
    

class GrammarConstraint(ABC):
    # Initialize the GrammarConstraint class with the model URI and tokenizer
    def __init__(self, model_uri, tokenizer):
        self.model_uri = model_uri
        self.tokenizer = tokenizer
        self.parser = None

    @abstractmethod
    def construct_filter_set(self, expression):
        pass


class PythonConstraint(GrammarConstraint): 

    # Construct a filter set for valid tokens based on a given expression
    def construct_filter_set(self, expression):
        vocab = self.tokenizer.vocab
        valid_tokens = []
        specials = ['*', '+', ')']

        # Preprocess expression to handle special cases
        if expression[0] in specials:
            expression = f'\{expression}'
        elif len(expression) == 1 and (expression == '[' or  expression == '('):
            expression = f'\{expression}'
        
        try:
            # Compile the regex pattern and use it to match valid tokens in the vocabulary
            pattern = re.compile(expression, re.UNICODE)
            for token, id in vocab.items():
                if pattern.match(token) is not None:
                   valid_tokens.append((token, id))
        except Exception as e: 
            print("Regex Compiling Error: ", f"{e} - {expression}") 
    
        # Return a set of valid tokens based on the pattern
        return set(valid_tokens)
    
    def parse_grammar(self):
        # Create the Python parser with Lark, using the LALR algorithm
        self.parser = Lark.open_from_package('lark', 'python.lark', ['grammars'], parser='lalr', regex=True, lexer='contextual', postlex=PythonIndenter(), start='file_input')
        terminals = self.parser.terminals
        t_map = {}
        for t in terminals:
            t_map[t.name] = t.pattern.value

        # Return a map of terminal tokens and their corresponding regex patterns
        return t_map

    def _prefix_state(self, prefix_str):
        valid_next = []
        try:      
            # Parse the token sequence
            interactive_tree = self.parser.parse_interactive(prefix_str)
            interactive_tree.exhaust_lexer()
            
            # Get the valid next states
            valid_next = list(interactive_tree.accepts())
        except Exception as e:
            print("Parsing Error: ", e)
        
        # Return a list of valid next terminals
        return valid_next

    def construct_final_filter_set(self, prefix_ids, terminals_map):
        # Decode the prefix IDs
        prefix_str = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)[0]
        
        # Get valid next terminals for the prefix
        next_lex = self._prefix_state(prefix_str)
        valid_next_ids = []
        for lex in next_lex:
            # Get the token set for each terminal symbol
            token_set = terminals_map[lex]
            for t in token_set:
                # Add valid token IDs to the list
                valid_next_ids.append(t[-1])
        
        # Return a set of valid next token IDs
        return set(valid_next_ids)


class JSONConstraint(GrammarConstraint): 

    # Construct a filter set for valid tokens based on a given expression
    def construct_filter_set(self, expression):
        vocab = self.tokenizer.vocab
        valid_tokens = []
        specials = ['*', '+', ')']

        # Preprocess expression to handle special cases
        if expression[0] in specials:
            expression = f'\{expression}'
        elif len(expression) == 1 and (expression == '[' or  expression == '('):
            expression = f'\{expression}'

        try:
            # Compile the regex pattern and use it to match all valid tokens in the vocabulary
            pattern = re.compile(expression, re.UNICODE)
            for token, id in vocab.items():
                if pattern.match(token) is not None:
                   valid_tokens.append((token, id))
        except Exception as e: 
            print(e, expression) 
    
        # Return a set of valid tokens based on the pattern
        return set(valid_tokens)
    
    def parse_grammar(self):
        # Define JSON grammar
        json_grammar = r"""
        ?start: value

        ?value: object
            | array
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null

        array  : "[" [value ("," value)*] "]"
        object : "{" [pair ("," pair)*] "}"
        pair   : string ":" value

        string : ESCAPED_STRING

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS

        %ignore WS
        """
        
        # Create Lark internal transformer to make parsing faster and more memory efficient 
        class TreeToJson(Transformer):
            @v_args(inline=True)
            def string(self, s):
                return s[1:-1].replace('\\"', '"')

            array = list
            pair = tuple
            object = dict
            number = v_args(inline=True)(float)

            null = lambda self, _: None
            true = lambda self, _: True
            false = lambda self, _: False


        # Create the JSON parser with Lark, using the LALR algorithm
        self.parser = Lark(json_grammar, parser='lalr',
                        lexer='contextual',
                        transformer=TreeToJson())
        terminals = self.parser.terminals
        t_map = {}
        for t in terminals:
            t_map[t.name] = t.pattern.value
        
        # Return a map of terminal tokens and their corresponding regex patterns
        return t_map

    def _prefix_state(self, prefix_str):
        valid_next = []
        try:      
            # Parse the token sequence
            interactive_tree = self.parser.parse_interactive(prefix_str)
            interactive_tree.exhaust_lexer()
            
            # Get the valid next states
            valid_next = list(interactive_tree.accepts())
        except Exception as e:
            print("Parsing Error: ", e)
        
        # Return a list of valid next terminals
        return valid_next
    
    def construct_final_filter_set(self, prefix_ids, terminals_map):
        # Decode the prefix IDs
        prefix_str = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)[0]
        
        # Get valid next terminals for the prefix
        next_lex = self._prefix_state(prefix_str)
        valid_next_ids = []
        for lex in next_lex:
            # Get the token set for each terminal symbol
            token_set = terminals_map[lex]
            for t in token_set:
                # Add valid token IDs to the list
                valid_next_ids.append(t[-1])
        
        # Return a set of valid next token IDs
        return set(valid_next_ids)


class GrammarLogitsProcessor(LogitsProcessor):

    def __init__(self, constraint_class, terminals_map):
        # Initialize the GrammarLogitsProcessor class with a constraint class and terminals map.
        self.constraint_class = constraint_class 
        self.terminals_map = terminals_map  

    def __call__(self, input_ids, scores):
        # This method is called for each generation step
        
        # Initialize a boolean bias tensor with the same shape as scores
        bias = torch.zeros_like(scores, dtype=torch.bool)
        
        # Get the set of valid next token IDs for current step
        valid_next_ids = self.constraint_class.construct_final_filter_set(input_ids, self.terminals_map)
        
        # Set the bias to True for valid next token IDs
        for id in valid_next_ids:
            bias[0, id] = True
        
        # Add the bias to the scores tensor to zero-out invalid next tokens
        scores += bias
        
        # Return the modified scores tensor
        return scores
