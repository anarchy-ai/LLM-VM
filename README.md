![Anarchy Logo](anarchy_logo.svg)

# 🚀 Anarchy LLM-VM 🚀

This is Anarchy's attempt at building 🏗️ generalized artificial intelligence 🤖 through the LLM-VM: a way to give your LLMs superpowers 🦸 and superspeed 🚄.

> This project is in BETA. Expect continuous improvement and development.

## About

### What is the Anarchy LLM-VM?

The Anarchy LLM-VM is a highly optimized and opinionated backend for running LLMs with all the modern features we've come to expect from completion: tool usage, persistent stateful memory, live data augmentation, data and task fine-tuning, output templating, a web playground, api endpoints, student-teacher distillation, data synthesis, load-balancing and orchestration, large context-window mimicry.

Formally, it is a virtual machine/interpreter for human language, coordinating between data, models (CPU), your prompts (code), and tools (IO). 

By doing all these things in one spot in an opinionated way, the LLM-VM can properly optimize and batch calls that would be exorbitantly expensive with distributed endpoints.  It furthermore strives for both model and architecture agnosticism, properly optimizing the chosen model for the current architecture.

### Why use the Anarchy LLM-VM?

In line with Anarchy's mission, the LLM-VM strives to support open-source models. By utilizing open-source models and running them locally you achieve a number of benefits:

* **Simplify your AI development:** *With AnarchyAI, one interface is all you need to interact with the latest LLMs available.*
  
* **Lower your costs:** *Running models locally can reduce the pay-as-you-go costs of development and testing.*
  
* **Unparalleled flexibility:** *AnarchyAI allows you to rapidly switch between models so you can pinpoint the exact right tool for your project.*
  
* **Strong community:** *Join our active community of highly motivated developers and engineers working passionately to democratize AI*
  
* **WYSIWYG:** *Open source means nothing is hidden; we strive for transparency and efficiency so you can focus on building.*

### How does Anarchy LLM-VM compare with other projects?

Anarchy is built by hackers, for hackers. Stemming from that principle, our product is designed to be simple, helpful, and flexible to apply to as many use cases as possible. *YOU* are the center of our business and we're working to make it as simple as running `pip3 install` to put the power of AI in *your* hands. 


* No other project grants as much freedom as Anarchy. We DGAF what you use our OSS AI for -- just don’t use our hosted playground for anything illegal.
  
* At Anarchy, we believe in open source and the power of community. Want a new feature? Come help us make it!
  
* Running your models on your hardware is key to democratizing AGI. We might be able to make it faster on ours, but every optimiztion made effects how these models behave and thus one’s ability to hack them. We’ll always keep this core open.


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

The code in `src/llm_vm/completion/` provides an optimizing completion library.  This technique intelligently analyzes call-site usage and automatically initiates student-teacher distillation to fine-tune purpose-specialized small and efficient models from slow and accurate general purpose models.

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
