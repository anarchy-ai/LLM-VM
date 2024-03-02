import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import floor


class SpeculativeSamplingWrapper:

    def __init__(self, device, **kwargs) -> None:
        # init parameters
        self.device = device
        self.draft_model_uri = kwargs["draft_model_uri"]

        self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_uri)
        self.draft_model.to(device)

        self.eos_token = self.draft_model.config.eos_token_id
        
        # hyperparameters
        self.k = kwargs.get("k", 5)
        self.scheduler = kwargs.get("scheduler", 2) # scheduler for k variable


    def canPerformSpeculativeSampling(self, target_model:AutoModelForCausalLM, target_tokenizer:AutoTokenizer) -> bool :
        """
        Checks if speculative sampling can be performed using the provided target model and tokenizer otherwise raises an exception.
        
        raise AssertionError if the use of speculative sampling was explicitly specified and one of the condition was not satisfied
        """
        draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_uri)

        assert self.draft_model is not None and target_model is not None, f"Only one model was provided, target model : {target_model} , draft model : {self.draft_model}"
        assert type(target_tokenizer) == type(draft_tokenizer), f"different tokenizers, target tokenizer : {type(target_tokenizer)} & draft tokenizer : {type(draft_tokenizer)}"
        assert self.draft_model.config.eos_token_id == target_model.config.eos_token_id, f"mismatch in eos token ids, {target_model.config.eos_token_id} and {self.draft_model.config.eos_token_id}"
        
        pad_id = target_model.config.pad_token_id
        target_tokenizer.pad_token_id = pad_id
        target_tokenizer.pad_token = "PAD"
        
        # speed up generation
        self.generation_kwargs = {
            "draft_model_key_values" : None,
            "past_key_values" : None,
        }
        return True
    
    # draft model generating the token
    def token_generation(self, ctx:torch.Tensor, limit:int, temperature:float, sample_func):
        all_probs = None
        for i in range(limit) :
            # cached key values
            past_keys_values = self.generation_kwargs.get("draft_model_key_values", None)

            # predict next token
            if past_keys_values is None :
                output = self.draft_model(ctx)
            else :
                actual_length = past_keys_values[0][0].shape[-2]
                new_length = ctx.shape[1] - actual_length
                # adding only the tokens which were not cached
                block = ctx[:, -new_length:]

                output = self.draft_model(block, past_key_values=past_keys_values)

            # model output processing
            logits = output.logits
            logits = logits / temperature if temperature else logits
            B, T, vocab_size = logits.shape
        
            probs = F.softmax(logits.view(T, vocab_size), dim=-1)

            # appending new token
            new_token = sample_func(probs[-1].view(1, -1))
            ctx = torch.cat([ctx, new_token], dim=-1)

            # saving the probabilities generated
            all_probs = torch.zeros((limit, probs.shape[1]), device=self.device) if all_probs is None else all_probs
            all_probs[i] = probs[-1].view(-1)
            
            self.generation_kwargs["draft_model_key_values"] = output.past_key_values

            if new_token.view(-1) == self.eos_token :
                break
        
        probs = all_probs[:i+1]
        return ctx, probs.view(-1, vocab_size)
    
    # target model generating the probabilities
    def probabilities_generation(self, ctx:torch.Tensor, target_model:AutoModelForCausalLM, temperature, sample_func) :
        # block size, past_keys, attention_mask
        model_inputs = target_model.prepare_inputs_for_generation(input_ids=ctx, **self.generation_kwargs)

        output = target_model(**model_inputs, output_attentions=False, output_hidden_states=False)
        # update generation kwargs with output
        self.generation_kwargs = target_model._update_model_kwargs_for_generation(output, self.generation_kwargs)
            
        logits = output.logits
        logits = logits / temperature if temperature else logits
        B, T, vocab_size = logits.shape
        
        probs = F.softmax(logits.view(T, vocab_size), dim=-1)

        new_token = sample_func(probs[-1].view(1, -1))

        ctx = torch.cat([ctx, new_token], dim=-1)

        return ctx, probs.view(-1, vocab_size)
    
    def crop_cached_key_values(self, key_values, num_token):
        trimmed_key_values = []
        for idx in range(len(key_values)):
            trimmed_key_values.append(
                (
                    key_values[idx][0][:, :, :num_token, :],
                    key_values[idx][1][:, :, :num_token, :],
                )
            )
        key_values = tuple(trimmed_key_values)
        return key_values
    
    def approval(self, ctx_token:torch.Tensor, draft_prob:torch.Tensor, target_prob:torch.Tensor, global_ctx:torch.Tensor, alignment:float, k:int, sample_func):
        # squeeze the batch dimension
        ctx_token = ctx_token.squeeze(0)
        # draft_prob = draft_prob.view(-1, self.vocab_size) # (T, vocab_size)
        # target_prob = target_prob.view(-1, self.vocab_size) # (T + 1, vocab_size)

        draft_token = ctx_token[:-1] # shape (window_size - 1)

        T = draft_token.shape[0]
        
        # approval_threshold = torch.rand((T)) # random values between (min_alignment, 1)
        # approval_threshold = torch.clip(approval_threshold, min_alignment, max=1).to(device)
        approval_threshold = torch.empty((T)).to(self.device)
        approval_threshold.fill_(alignment)
        
        # getting the probabilities for the selected tokens -> shape(B, T)
        draft_sampled_prob = torch.gather(draft_prob, dim=1, index=draft_token.unsqueeze(-1)) 
        target_sampled_prob = torch.gather(target_prob[:-1, :], dim=1, index=draft_token.unsqueeze(-1)) # we remove 1 bc no prob for last token
        # checking the divergence between draft and target
        prob_divergence = (target_sampled_prob / draft_sampled_prob)
        prob_divergence = torch.min(torch.ones(1, device=self.device), prob_divergence).view(T) # shape (T)
        
        # point of failure
        failure_mask = approval_threshold > prob_divergence 
        failure_mask = failure_mask.to(torch.int16) 
        index_failure = torch.argmax(failure_mask, dim=-1)

        # all tokens were valid, we get k + 1 tokens
        if failure_mask[index_failure] == False :
            global_ctx = torch.cat([global_ctx, ctx_token.view(1, -1)], dim=-1)
            # scheduler update
            k += floor(k * self.scheduler) if self.scheduler else 0
            return global_ctx, k
        
        # creating a distribution in the safe area (google deepmind paper)
        else :
            # reshaping the draft prob
            if target_prob.shape[-1] >= draft_prob.shape[-1] :
                draft_dist = torch.zeros((target_prob.shape[-1]), device=self.device)
                draft_dist[:draft_prob.shape[-1]] = draft_prob[index_failure] 
            else :
                draft_dist = torch.zeros((target_prob.shape[-1]), device=self.device)
                draft_dist[:] = draft_prob[index_failure][:target_prob.shape[-1]] 

            dist = target_prob[index_failure] - draft_dist

            dist = torch.max(torch.zeros((1), device=self.device), dist)
            dist = dist / (torch.sum(dist) or dist.shape[1]) # normalizing
            # sampling and assigning new token
            new_token = sample_func(dist.view(1, -1))[0]

        # scheduler update
        draft_token[index_failure] = new_token
        k = max(1, k - floor((k - index_failure - 1) // self.scheduler)) if self.scheduler else k

        # adding it 
        global_ctx = torch.cat([global_ctx, draft_token[:index_failure+1].view(1, -1)], dim=-1)
        num_tokens = global_ctx.shape[1]

        # crop the cached key values
        small_model_kv = self.generation_kwargs["draft_model_key_values"]
        big_model_kv = self.generation_kwargs["past_key_values"]

        self.generation_kwargs["draft_model_key_values"] = self.crop_cached_key_values(small_model_kv, num_tokens - 2) if small_model_kv is not None else None
        self.generation_kwargs["past_key_values"] = self.crop_cached_key_values(big_model_kv, num_tokens - 1) if big_model_kv is not None else None
        
        return global_ctx, k

    
    def complete(self, inputs:torch.Tensor, max_len:int, target_model:AutoModelForCausalLM, **kwargs):
        """Generate new tokens using speculative sampling algorithm"""
        
        temperature = kwargs.get("temperature", None)
        alignment = kwargs.get("alignment", 1)
        do_sample = kwargs.get("do_sample", False)
        max_new_tokens = max_len + inputs.shape[-1]

        self.generation_kwargs = {
            "draft_model_key_values" : None,
            "past_key_values" : None,
        }
        k = self.k

        # we use the implementation at token level from hugging face
        if alignment == 1 :
            if "alignment" in kwargs : del kwargs["alignment"]
            kwargs["max_new_tokens"] = max_len

            global_ctx = target_model.generate(inputs, assistant_model=self.draft_model, **kwargs)
            return global_ctx

        # device and function set up
        self.device = inputs.device

        if do_sample : sample_func = lambda x : torch.multinomial(x, num_samples=1).view(-1, 1)
        else : sample_func = lambda x : torch.argmax(x, dim=-1).view(-1, 1)
       
        global_ctx = inputs 
        while global_ctx.shape[1] < max_new_tokens  :
            ctx = global_ctx.clone()
            # small model prediction
            limit = k if global_ctx.shape[1] + k <= max_new_tokens else max_new_tokens - global_ctx.shape[1]
            ctx, draft_prob = self.token_generation(ctx, limit, temperature, sample_func)
            # we make a parallel prediction from the big model
            ctx, target_prob = self.probabilities_generation(ctx, target_model, temperature, sample_func)

            # we check the validity of the last predicted set
            num_new_token = draft_prob.shape[0]
            token_target_prob = target_prob[-num_new_token -1:]
            draft_pred_tokens = ctx[: , -num_new_token - 1:]
            token_draft_prob = draft_prob[-num_new_token:]

            # validating the tokens
            global_ctx, k = self.approval(draft_pred_tokens, token_draft_prob, token_target_prob, global_ctx, alignment, k, sample_func)
            
            # end of sentence verification
            if global_ctx[0, -1] == self.eos_token :
                break
          


        return global_ctx