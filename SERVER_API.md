### 🏃‍♀️ Running Standalone

After you have installed it (as above), you now have an anarchy server that provides a completion API (using Flask).

```bash
> cd LLM-VM
> llm_vm_server
```

This will start a flask server at https://localhost:3002/ and will create an endpoint http://localhost:3002/v1/complete.


## 🕸️ API Usage 🕸️

### Submitting a Request

The package defaults to using chatGPT as the big model that we use for completion, and GPT3 is the small model that we use for fine-tuning. These can be adjusted in src/llm_vm/completion/config.json.

#### Request Parameters
- `prompt`: String - The natural language query that is passed to the LLM
- `context`: String - The context for the prompt. We fine-tune models with respect to the context window. Can be empty
- `openai_key`: String - your open.ai key. You can generate one [here](https://platform.openai.com/account/api-keys)
- `temperature`?: Number - The temperature of the model. A higher temperature offers a more diverse range of answers, but a higher chance of straying from the context. 
- `finetune`?: Boolean - True if you want to finetune the model and False if not.
- `data_synthesis`?: Boolean - True if you want to finetune the model and False if not.
- `stoptoken`?: String or Array of String - Sequence of tokens that will stop generation and return the result up to and before the sequence provided.
- `tools`?: JSON Array
    - `description`: String - description of what the tool does.
    - `url`: String: Endpoint - of the tool you want to use
    - `dynamic_params`: JSON {"key": "value"} - Parameters that change with each API call, like a stock ticker symbol you are interested in
    - `method`: String - GET/POST
    - `static_params`: JSON {"key": "value"} - Parameters that are the same for every API call, like the token or API key


#### Submitting a Request to the LLM

Post requests can be sent to this endpoint `http://localhost:3002/v1/complete` in the following format:

```
{
    "prompt": "What is 2+2",
    "context": "Answer the math problem with just a number.",
    "openai_key": <OPENAI-KEY>,
    "temperature":0,
    "finetune": true,
    "data_synthesis: true
}

```
Example Curl:
```bash
curl -X POST http://localhost:3002/v1/complete -H "Content-Type: application/json" -d '{"prompt": "What is 2+2", "context": "Answer the math problem with just a number.", "openai_key": "your-key", "temperature": 0, "finetune": true, "data_synthesis": true}'
```

If no `temperature`, `finetune`, or `data_synthesis` are provided, these values will default to `0`, `false`, and `false` respectively. 

## 🛠️ Tool Using Agents 🛠️


#### Submitting a Request to the REBEL Agent

An agent is an algorithm that uses the LLM in a way to increase its reasoning capabilities such as allowing the use of outside tools or allowing the LLM to answer compositional questions. To use the REBEL (REcursion Based Extensible Llm) endpoint with tools, add a tool in the following way. We have used the finnhub API to get stock data as an example.

Post requests can be sent to this endpoint `http://localhost:3002/v1/complete` with the tools optional parameter:
```
{
    "prompt": "What is the price of apple stock?",
    "context":"",
    "tools":[{
        "description": "Find the current price of a stock.",
        "url": "https://finnhub.io/api/v1/quote",
        "dynamic_params":{"symbol": "the symbol of the stock"},
        "method": "GET",
        "static_params":{"token":<Finnhub Key>}
    }],
    "openai_key": <OpenAI Key>,
    "temperature":0,
    "finetune": true
}

```

To access the completion endpoint programmatically, after starting up the server on your machine, import llm_vm.py into your code and call the completion function. Currently, we do not support tools through the programmatic completion function. To test that the server is working correctly, after starting it, run 
```python 
test_llm_vm.py
```
