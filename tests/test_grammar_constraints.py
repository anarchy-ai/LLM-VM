from llm_vm.guided_completion import GrammarCompletion
from transformers import AutoTokenizer

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", padding_side='left')
    model = GrammarCompletion("gpt2-medium", tokenizer)

    # For python
    prompt = "rand_num = 5"
    result = model.complete(prompt, temperature=1.0, max_new_tokens=20)
    print(result)

    # for JSON
    # prompt = "{ \"name\": \"Vik\", \"age\": \"50\","
    # result = model.complete(prompt, grammar_type="json", temperature=1.0, max_new_tokens=20)
    # print(result)
