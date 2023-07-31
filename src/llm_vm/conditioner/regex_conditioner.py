import re
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import numpy as np
from transformers import LogitsProcessor

'''
Copied and adapted from: https://github.dev/r2d4/rellm/blob/main/rellm
'''



def _is_valid_regex(text):
    try:
        re.compile(text)
        return True 
    except re.error:
        return False
    

class TokenFilter:
    def __init__(self , tokenizer) -> None:
        self.tokenizer = tokenizer
        self._build_decoded_cache()

    def _decode_token(self , token_id):
        return token_id , self.tokenizer.decode(token_id)

    def _build_decoded_cache(self):
        vocab_items = self.tokenizer.get_vocab().items()
        num_threads = multiprocessing.cpu_count()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            decoded_token = dict(executor.map(lambda x : self._decode_token(x[1]) , vocab_items))

        self.decoded_token_cache = decoded_token

    def _is_valid_token(self , token_id , partial_completion , pattern):
        decoded_token = self.decoded_token_cache[token_id]
        return pattern.fullmatch(partial_completion + decoded_token)

    def filter_tokens(self , partial_completion , pattern):
        num_threads = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=num_threads):
            valid_token_ids = set(
                filter(
                lambda token_id : self._is_valid_token(token_id , partial_completion , pattern) , 
                self.decoded_token_cache.keys()
                )
            )        
        return valid_token_ids
    


class LogitsMask(LogitsProcessor):
    """
    LogitsMask is a LogitsProcessor that masks logits for tokens that are 
    not in the allowed token ids set.
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = np.ones_like(scores) * -1e10
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask 
        return scores



class NoMaskProcessor(LogitsProcessor):
    
    def __init__(self):
        pass
    def __call__(self, input_ids, scores):
        return scores




class RegexConditioner:
    def __init__(self , text, tokenizer) -> None:
        assert _is_valid_regex(text=text) , f"{text} is not a valid regex"
        self.regex = re.compile(text)
        self.token_filter = TokenFilter(tokenizer=tokenizer)
        

    
    def filter_and_mask_tokens(self , partial_completion):
        allowed_token_ids = self.token_filter.filter_tokens(partial_completion, self.regex)
        custom_mask_processor = LogitsMask(allowed_token_ids)

        return custom_mask_processor
        
            
