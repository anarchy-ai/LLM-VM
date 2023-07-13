import os
import llm_vm.client as l

def basic_completion_test():
    data = l.complete(
        prompt = 'How long does it take for an apple to grow?',
        context = '',
        openai_key = os.getenv("OPENAI_API_KEY"))
    return data

def stop_token_test():
    data = l.complete(
        prompt = 'How do you make butter?',
        context = 'repeat the prompt 5 times',
        openai_key = os.getenv("OPENAI_API_KEY"),
        stoptoken = 'butter')
    return data

def stop_tokens_test():
    data = l.complete(
        prompt = 'How do you make butter?',
        context = 'repeat the prompt 5 times',
        openai_key = os.getenv("OPENAI_API_KEY"),
        stoptoken = ['butter', 'batter', 'bitter', 'bread'])
    return data

if __name__ == '__main__':
    print(basic_completion_test())
    # print(stop_token_test())
    # print(stop_tokens_test())
