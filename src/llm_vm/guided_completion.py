import outlines.models as models
import outlines.text.generate as generate
import torch
from lark import Lark, Transformer, v_args
from lark.indenter import PythonIndenter
from transformers import (AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor, AutoTokenizer)
import re
from abc import ABC,abstractmethod

model = models.transformers("gpt2")


class Completion(ABC):
    """
    A class used to generate completions when optimize.complete is called
    """

    @abstractmethod
    def complete(self, prompt):
        pass

    @staticmethod
    def create(regex, type, choices, grammar_type, *, default=None):
        completion = default
        if regex is not None:
            completion = GenerativeCompletion.regex_completion(regex)
        elif type is not None:
            completion = GenerativeCompletion.type_completion(type)
        elif choices is not None:
            completion = GenerativeCompletion.choices_completion(choices)
        elif grammar_type is not None:
            tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", padding_side='left')
            completion = GrammarCompletion("gpt2-medium", tokenizer)
        return completion


class GenerativeCompletion(Completion):
    def __init__(self, generator, *generator_args):
        """
        Parameters: 
        -----------
        
        generator (Callable[[Transformers, ...generator_args], None]): Generator function to be used on the complete
        *generator_args (Any): Generator arguments (without model)
        
        """
        self.generator = generator
        self.generator_args = generator_args

    def complete(self, prompt):
        return self.generator(model, *self.generator_args)(prompt)

    @staticmethod
    def regex_completion(regex):
        return GenerativeCompletion(generate.regex, regex)

    @staticmethod
    def choices_completion(choices):
        return GenerativeCompletion(generate.choice, choices)

    @staticmethod
    def type_completion(type_name):
        if type_name not in ["float", "integer"]:
            raise Exception("type must be float or integer")
        return GenerativeCompletion(getattr(generate, type_name))

    @staticmethod
    def response_completion():
        return GenerativeCompletion(lambda _: (lambda x: x['response']))


class GrammarCompletion(Completion):
    def __init__(self, model_uri, tokenizer, grammar_type='python'):
        # Initialize the GrammarCompletion class with a model URI and tokenizer
        self.model_identifier = model_uri
        self.tokenizer = tokenizer

        # Load model from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(self.model_identifier)
        print(f"{self.model_identifier} model is ready for use on {self.model.device}", flush=True)
        
        self.constraint = GrammarConstraint.create(grammar_type, model_uri, tokenizer)

        # Initialize dict to store terminal symbols and their corresponding token mappings
        self.terminals_tokens_map = {}

    @property
    def eos_token_id(self):
        return self.model.config.eos_token_id
    
    def complete(self, prompt, **model_kwargs):
        # Check if the specified grammar type is supported

        # Encode the input prompt
        token_ids = self.tokenizer(prompt, return_tensors='pt')
        input_len = len(token_ids['input_ids'][-1])

        # Construct terminal tokens mapping if it's not yet initialized
        if len(self.terminals_tokens_map) == 0:
            terminal_regex_map = self.constraint.parse_grammar()
            for k, v in terminal_regex_map.items():
                matching_tokens = self.constraint.construct_filter_set(v)
                self.terminals_tokens_map[k] = matching_tokens

            # Add end-of-sequence token mapping
            self.terminals_tokens_map['$END'] = [(self.tokenizer.decode(self.eos_token_id), self.eos_token_id)]

        # Initialize the LogitsProcessorList for processing sequence logits and append GrammarLogitsProcessor instance
        logits_processor = LogitsProcessorList()
        logits_processor.append(GrammarLogitsProcessor(self.constraint, self.terminals_tokens_map))

        model_kwargs["do_sample"] = model_kwargs["temperature"] > 0
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
        self._parser_state = None
        self._copy_state = False

    @abstractmethod
    def construct_filter_set(self, expression):
        pass

    @abstractmethod
    def parse_grammar(self):
        pass

    @staticmethod
    def create(grammar_type, model_uri, tokenizer):
        grammar_type = grammar_type.lower()
        grammars = {
            'python': PythonConstraint,
            'json': JSONConstraint,
        }

        if grammar_type not in grammars.keys():
            raise ValueError(f'{grammar_type} is not supported. The only valid grammar types are {grammars.keys()}')

        return grammars[grammar_type](model_uri, tokenizer)


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

    def _prefix_state(self, prefix_str=None, last_token=None):
        valid_next = []
        if self._parser_state is None:
            try:      
                # Parse the entire token sequence
                interactive_tree = self.parser.parse_interactive(prefix_str)
                interactive_tree.exhaust_lexer()
                self._parser_state = interactive_tree.copy()
                
                # Get the valid next states
                valid_next = list(interactive_tree.accepts())
            except Exception as e:
                # print("Parsing Error: ", e)
                pass
        else:
            try:
                # lex the last token
                last_lex = list(self.parser.lex(last_token))
                # update parser state by feeding last token
                self._parser_state.feed_token(last_lex[0])
                valid_next = list(self._parser_state.accepts())
            except Exception as e:
                # print("Parsing Error: ", e)
                pass

        # Return a list of valid next terminals
        return valid_next

    def construct_final_filter_set(self, prefix_ids, terminals_map):
        valid_next_ids = []
        
        if self._copy_state == True:
            # Decode only the last prefix ID
            last_token = self.tokenizer.batch_decode(prefix_ids[:, -1], skip_special_tokens=True)[0]
            # Get valid next terminals from the last token
            next_lex = self._prefix_state(last_token=last_token)
        else:
            # Decode all the prefix IDs
            prefix_str = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)[0]
            # Get valid next terminals for the prefix
            next_lex = self._prefix_state(prefix_str=prefix_str)
            self._copy_state = True

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

    def _prefix_state(self, prefix_str=None, last_token=None):
        valid_next = []
        if self._parser_state is None:
            try:      
                # Parse the entire token sequence
                interactive_tree = self.parser.parse_interactive(prefix_str)
                interactive_tree.exhaust_lexer()
                self._parser_state = interactive_tree.copy()
                
                # Get the valid next states
                valid_next = list(interactive_tree.accepts())
            except Exception as e:
                # print("Parsing Error: ", e)
                pass
        else:
            try:
                # lex the last token
                last_lex = list(self.parser.lex(last_token))
                # update parser state by feeding last token
                self._parser_state.feed_token(last_lex[0])
                valid_next = list(self._parser_state.accepts())
            except Exception as e:
                # print("Parsing Error: ", e)
                pass

        # Return a list of valid next terminals
        return valid_next

    def construct_final_filter_set(self, prefix_ids, terminals_map):
        valid_next_ids = []
        
        if self._copy_state == True:
            # Decode only the last prefix ID
            last_token = self.tokenizer.batch_decode(prefix_ids[:, -1], skip_special_tokens=True)[0]
            # Get valid next terminals from the last token
            next_lex = self._prefix_state(last_token=last_token)
        else:
            # Decode all the prefix IDs
            prefix_str = self.tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)[0]
            # Get valid next terminals for the prefix
            next_lex = self._prefix_state(prefix_str=prefix_str)
            self._copy_state = True

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
