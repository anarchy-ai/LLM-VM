# LLM-VM

This is an LLM agnostic JIT for natural language. Specifically, it uses LLMs to convert conversational natural language into a dynamic series of LLM and IO commands. You provide the underlying provider(s), actions (APIs, code-hooks) and their descriptions, data-sources (PDFs, websites...), and the LLM-VM will take care of load-balancing, fine-tuning, natural language compilation and tool-selection.

This is still in BETA.  Very little attention has been paid to package structure.  Expect it to change.

## Running/Testing

### Installation and Starting the Server

To run and test this repository you need to start a flask server. To start the flask server use: 

```bash
pip3 install .
llm_vm_server
```

This will start a flask server at https://localhost:3002/ and will create an endpoint https://localhost:3002/v1/completion. 

### Submitting a Request

The package defaults to using chatGPT as the big model that we use for completion, and GPT3 is the small model that we use for fine tuning. These can be adjusted in src/llm_vm/completion/config.json.

#### Request Paramaters
- `prompt`: String - The natural language query that is passed to the LLM
- `context`: String - The context for the prompt. We fine tune models with respect to the context window. Can be empty
- `openai_key`: String - your personal open.ai key. You can generate one [here](https://platform.openai.com/account/api-keys)
- `temperature`?: Number - The temperature of the model. A higher temperature offers a more diverse range of answers, but a higher chance of straying from the context. 
- `finetune`?: Boolean - True if you want to finetune the model and False if not.
- `data_synthesis`?: Boolean - True if you want to finetune the model and False if not.
- `tools`?: JSON Array
	- `description`: String - description of what the tool does.
	- `url`: String: Endpoint - of the tool you want to use
	- `dynamic_params`: JSON {"key":"value"} - Parameters that change with each api call, like a stock ticker symbol you are interested in
	- `method`: String - GET/POST
	- `static_params`: JSON {"key":"value"} - Parameters that are the same for every api call, like the token or API key


#### Submitting a Request to the LLM

Post requests can be sent to this endpoint `https://localhost:3002/` in the following format:

```
{
    "prompt":"What is 2+2",
    "context":"Answer the math problem with just a number.",
    "openai_key": <OPENAI-KEY>,
    "temperature":0,
    "finetune": true,
    "data_synthesis: true
}
```

If no `temperature`, `finetune`, or `data_synthesis` are provided, these values will default to `0`, `false`, and `false` respectively. 

#### Submitting a Request to the REBEL Agent

An agent is an algorithm that uses the LLM in a way to increase its reasoning capabilities such as allowing the use of outside tools or allowing the LLM to answer compositional questions. To use the REBEL (REcursion Based Extensible Llm) endpoint with tools, add a tool in the following way. We have used the finnhub api to get stocks data as an example.

Post requests can be sent to this endpoint `https://localhost:3002/` with the tools optional parameter:
```
{
    "prompt":"What is the price of apple stock?",
    "context":"",
    "tools":[{
        "description":"Find the current price of a stock.",
        "url":"https://finnhub.io/api/v1/quote",
        "dynamic_params":{"symbol": "the symbol of the stock"},
        "method":"GET",
        "static_params":{"token":<Finnhub Key>}
    }],
    "openai_key": <OpenAI Key>,
    "temperature":0,
    "finetune": true
}

```

To access the completion endpoint programmatically, after starting up the server on your machine, import llm_vm.py into your code and call the completion function. Currently, we do not support tools through the programmatic completion function. To test that the server is working correctly, after starting it, run 
```
python test_llm_vm.py
```
### Optimizing Text Generation

The code in `src/llm_vm/completion/` provides an optimizing text generation framework. It allows storing models and fine-tuning them on new data to improve generations over time.

#### Files

- `optimize.py` - Provides the `Optimizer` abstract class and implementations for local optimization (using a `local_ephemeral` store) and hosted optimization (using OpenAI's API).
- `test_optimize.py` - A test file that shows using either the local or hosted optimizer to generate Haskell code and improve over time.

#### Usage

To use the local optimizer:

```python
from optimize import LocalOptimizer

optimizer = LocalOptimizer(MIN_TRAIN_EXS=2)  # Require 2 examples before fine-tuning a new model

completion = optimizer.complete(
    "Please convert this line to some haskell:", # Description of the task
    "x = 5",   # Prompt to complete
    max_tokens=100, 
    temperature=0.0
)
print(completion)
# Haskell:
# x = 5
```

To use the hosted optimizer:

```python
from optimize import HostedOptimizer
from dotenv import load_dotenv

load_dotenv()

optimizer = HostedOptimizer(anarchy_key=os.getenv('ANARCHY_KEY'), openai_key=os.getenv('OPENAI_KEY'), MIN_TRAIN_EXS=2)  

completion = optimizer.complete(
    "Please convert this line to some haskell:", 
    "x = 5",
    max_tokens=100, 
    temperature=0.0
)
print(completion)
# Haskell:  
# x = 5
```

There are three agents: FLAT, REBEL, and BACKWARD_CHAINING. 

Run the agents separately by going into the `src/llm_vm/agents/<AGENT_FOLDER>` and running the file that is 
titled `agent.py`. 

Alternatively, to run a simple interface and choose an agent to run from the CLI, run the `src/llm_vm/agents/agent_interface.py` file 
and follow the command prompt instructions. 

## License

`llm_vm` was created by Matthew Mirman. It is licensed under the terms of the MIT license.
