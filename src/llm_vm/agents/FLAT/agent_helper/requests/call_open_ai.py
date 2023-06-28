import openai
import os
from typings import *

def __tokens_to_dollars(usage) -> float:
    return float(usage)/1000*0.02

def call_open_ai(request: LLMCallParams) -> LLMCallReturnType:
    model = request['model']
    max_tokens = request['max_tokens']
    stop = request['stop'] if 'stop' in request else None
    cur_prompt = request['prompt']
    temperature = request['temperature'] if 'temperature' in request else 0.0
    chat = request['llm']
    
    if isinstance(model, tuple):
        model, api_key = model[0], model[1]
    else:
        api_key = False
    
    current_key = None
    if api_key:
        current_key = openai.api_key
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
        
    if chat == LLMCallType.OPENAI_CHAT:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                max_tokens=max_tokens,
                stop=stop,
                messages=cur_prompt,
                temperature=temperature
            )
        response_text = response['choices'][0]['message']['content']
    elif chat == LLMCallType.OPENAI_COMPLETION:

        response = openai.Completion.create(
                model=model,
                max_tokens=max_tokens,
                stop=stop,
                prompt=cur_prompt,
                temperature=temperature
            )

        response_text = response['choices'][0]['text']
        
    if current_key:
        openai.api_key = current_key
        os.environ["OPENAI_API_KEY"] = current_key
         
    usage = response["usage"]["total_tokens"]
    price = __tokens_to_dollars(usage)
    return response_text.strip("\n "), price