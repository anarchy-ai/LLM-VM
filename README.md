![Anarchy Logo](anarchy_logo.svg)
Need help? Want to help out? Come join our discord server! Our engineers are 
standing by to help out!  https://discord.gg/YmNvCAk6W6


# 🤖 Anarchy LLM-VM 🤖
*Simplifying AGIs for accelerating development*

This is [Anarchy's](https://anarchy.ai) effort for building 🏗️ open generalized artificial intelligence 🤖 through the LLM-VM: a way to give your LLMs superpowers 🦸 and superspeed 🚄. 

You can find instructions to try it live here: [anarchy.ai](https://anarchy.ai)

> This project is in BETA. Expect continuous improvement and development.


# Table of Contents

* [Table of Contents](#Table)
* [About](#-About-)
    * [What](#-what-is-the-anarchy-llm-vm)
    * [Why](#-why-use-the-anarchy-llm-vm)
    * [Features and Roadmap](#-features-and-roadmap)
* [Quick Start and Installation](#-quickstart-)




## 📚 About 📚

### 💁 What is the Anarchy LLM-VM?

The Anarchy LLM-VM is a highly optimized and opinionated backend for running LLMs with all the modern features we've come to expect from completion: tool usage, persistent stateful memory, live data augmentation, data and task fine-tuning, output templating, a web playground, api endpoints, student-teacher distillation, data synthesis, load-balancing and orchestration, large context-window mimicry.

Formally, it is a virtual machine/interpreter for human language, coordinating between data, models (CPU), your prompts (code), and tools (IO). 

By doing all these things in one spot in an opinionated way, the LLM-VM can properly optimize and batch calls that would be exorbitantly expensive with distributed endpoints.  It furthermore strives for both model and architecture agnosticism, properly optimizing the chosen model for the current architecture.

### 🤌 Why use the Anarchy LLM-VM?

In line with Anarchy's mission, the LLM-VM strives to support open-source models. By utilizing open-source models and running them locally you achieve a number of benefits:

* **Speed up your AGI development 🚀:** *With AnarchyAI, one interface is all you need to interact with the latest LLMs available.*
  
* **Lower your costs 💸:** *Running models locally can reduce the pay-as-you-go costs of development and testing.*
  
* **Flexibility 🧘‍♀️:** *AnarchyAI allows you to rapidly switch between models so you can pinpoint the exact right tool for your project.*
  
* **Community Vibes 🫂:** *Join our active community of highly motivated developers and engineers working passionately to democratize AGI*
  
* **WYSIWYG 👀:** *Open source means nothing is hidden; we strive for transparency and efficiency so you can focus on building.*

### 🎁 Features and Roadmap

* **Implicit Agents 🔧🕵️:** *The Anarchy LLM-VM can be set up to use external tools through our agents such as **REBEL** just by supplying tool descriptions!*

* **Inference Optimization 🚄:** *The Anarchy LLM-VM is optimized from agent level all the way to assembly on known LLM architectures to get the most bang for your buck. With state of the art batching, sparse inference and quantization, distillation, and multi-level colocation, we aim to provide the fastest framework available.*

* **Task Auto-Optimization 🚅:** *The Anarchy LLM-VM will analyze your use cases for repetative tasks where it can activate student-teacher distillation to train a super-efficient small model from a larger more general model without loosing accuracy.  It can furthermore take advantage of data-synthesis techniques to improve results.*

* **HTTP Endpoints 🕸️:** *We provide an HTTP standalone server to handle completion requests.*

* **Library Callable 📚:** *We provide a library that can be used from any python codebase directly.*

* **Live Data Augmentation 📊:** (ROADMAP) *You will be able to provide a live updating data-set and the Anarchy LLM-VM will **fine-tune** your models or work with a **vector DB** to provide up-to-date information with citations*

* **Web Playground 🛝:** (ROADMAP) *You will be able to run the Anarchy LLM-VM and test it's outputs from the browser.*

* **Load-Balancing and Orchestration ⚖️:** (ROADMAP) *If you have multiple LLMs or providers you'd like to utilize, you will be able to hand them to the Anarchy LLM-VM to automatically figure out which to work with and when to optimize your uptime or your costs*

* **Output Templating 🤵:** (ROADMAP) *You can ensure that the LLM only outputs data in specific formats and fills in variables from a template with either regular expressions, LMQL, or OpenAI's template language*

* **Persistent Stateful Memory 📝:** (ROADMAP) *The Anarchy LLM-VM can remember a user's conversation history and react accordingly*

## 🚀 Quickstart 🚀

### 👨‍💻 Installation

To install the LLM-VM you simply need to download this repository and install it with pip like so:

```bash
> git clone https://github.com/anarchy-ai/LLM-VM.git
> cd LLM-VM
> pip3 install .
```

This will install both the library and test-server.  

### Generating Completions
Our LLM-VM gets you working directly with popular LLMs locally in just 3 lines. Once you've installed (as above), just load your model and start generating!

```python
# import our client
from llm_vm.client import Client

# Select which LLM you want to use, here we have openAI's 
client = Client(big_model = 'chat_gpt')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '', openai_key = 'ENTER_YOUR_API_KEY')
print(response)
# Anarchy is a political ideology that advocates for the absence of government...
```

### Locally Run an LLM
```python
# import our client
from llm_vm.client import Client

# Select the LlaMA model
client = Client(big_model = 'llama')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```


### Supported Models
Select from the following models
```python
Supported_Models = ['chat_gpt','gpt','neo','llama','bloom']
```




### Picking a Different Model
LLM-VM default model sizes for local models is intended to make experimentation 
with LLMs accessible to everyone, but if you have the memory required, larger parameter models 
will perform far better!

for example if you want to use a large and small neo model  for your teacher and student,


```python
# import our client
from llm_vm.client import Client

# Select the LlaMA model
client = Client(big_model = 'neo',big_model_config={'uri_override'})

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```


#### Neo Model 
 | URI | Model Params | Checkpoint file size | Is Default?
 -----------------------------------------------------------
| EleutherAI/gpt-neo-125m | 125m | 526 MB | ❌
| EleutherAI/gpt-neo-1.3B | 1.3B | 5.31 GB | ✅
| EleutherAI/gpt-neo-2.7B | 2.7B | 10.7 GB | ❌
| EleutherAI/gpt-neox-20b | 20B | 41.3 GB  | ❌

#### Bloom Model  


||


### 🏃‍♀️ Running Standalone

After you have installed (as above), you now have an anarchy server which provides an completion API (using flask).

```bash
> cd LLM-VM
> llm_vm_server
```

This will start a flask server at https://localhost:3002/ and will create an endpoint https://localhost:3002/v1/completion.

## 🕸️ API Usage 🕸️

### Submitting a Request

The package defaults to using chatGPT as the big model that we use for completion, and GPT3 is the small model that we use for fine tuning. These can be adjusted in src/llm_vm/completion/config.json.

#### Request Paramaters
- `prompt`: String - The natural language query that is passed to the LLM
- `context`: String - The context for the prompt. We fine tune models with respect to the context window. Can be empty
- `openai_key`: String - your personal open.ai key. You can generate one [here](https://platform.openai.com/account/api-keys)
- `temperature`?: Number - The temperature of the model. A higher temperature offers a more diverse range of answers, but a higher chance of straying from the context. 
- `finetune`?: Boolean - True if you want to finetune the model and False if not.
- `data_synthesis`?: Boolean - True if you want to finetune the model and False if not.
- `stoptoken`?: String or Array of String - Sequence of tokens that will stop generation and return the result up to and before the sequence provided.
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

## 🛠️ Tool Using Agents 🛠️


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

## 🚅 Optimizing Completion 🚅

The code in `src/llm_vm/completion/` provides an optimizing completion library.  This technique intelligently analyzes call-site usage and automatically initiates student-teacher distillation to fine-tune purpose-specialized small and efficient models from slow and accurate general purpose models.


#### Files

- `optimize.py` - Provides the `Optimizer` abstract class and implementations for local optimization (using a `local_ephemeral` store) and hosted optimization (using OpenAI's API).
- `test_optimize.py` - A test file that shows using either the local or hosted optimizer to generate Haskell code and improve over time.

#### Usage

There are two agents: FLAT and REBEL. 

Run the agents separately by going into the `src/llm_vm/agents/<AGENT_FOLDER>` and running the file that is 
titled `agent.py`. 

Alternatively, to run a simple interface and choose an agent to run from the CLI, run the `src/llm_vm/agents/agent_interface.py` file 
and follow the command prompt instructions. 

## Acknowledgements 
Matthew Mirman, Abhirgya Sodani, Carter Schonwald, Andrew Nelson


## License

`llm_vm` was created by Matthew Mirman. It is licensed under the terms of the MIT license.
