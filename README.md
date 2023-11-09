![Anarchy Logo](https://github.com/VictorOdede/LLM-VM/raw/main/anarchy_logo.svg)

<p align="center">
  <a href="https://anarchy.ai/" target="_blank"><img src="https://img.shields.io/badge/View%20Documentation-Docs-yellow"></a>
  <a href="https://discord.gg/YmNvCAk6W6" target="_blank"><img src="https://img.shields.io/badge/Join%20our%20community-Discord-blue"></a>
  <a href="https://github.com/anarchy-ai/LLM-VM">
      <img src="https://img.shields.io/github/stars/anarchy-ai/LLM-VM" />
  </a>
</p>
<h1 align='center'> 🤖 Anarchy LLM-VM 🤖 </h1>
<p align='center'><em>An Open-Source AGI Server for Open-Source LLMs</em></p>

This is [Anarchy's](https://anarchy.ai) effort to build 🏗️ an open generalized artificial intelligence 🤖 through the LLM-VM: a way to give your LLMs superpowers 🦸 and superspeed 🚄.

You can find detailed instructions to try it live here: [anarchy.ai](https://anarchy.ai)

> This project is in BETA. Expect continuous improvement and development.


# Table of Contents

* [Table of Contents](#table)
* [About](#-about-)
    * [What](#-what-is-the-anarchy-llm-vm)
    * [Why](#-why-use-the-anarchy-llm-vm)
    * [Features and Roadmap](#-features-and-roadmap)
* [Quick Start and Installation](#-quickstart-)
   * [Requirements](#-requirements)
   * [Installation](#-installation)
   * [Generating Completions](#-generating-completions)
   * [Running LLMs Locally](#-running-llms-locally)
   * [Supported Models](#-supported-models)
   * [Picking Different Models](#-picking-different-models)
   * [Tool Usage](#-tool-usage)
* [Contributing](#-contributing-)

## 📚 About 📚

### 💁 What is the Anarchy LLM-VM?

The Anarchy LLM-VM is a highly optimized and opinionated backend for running LLMs with all the modern features we've come to expect from completion: tool usage, persistent stateful memory, live data augmentation, data and task fine-tuning, output templating, a web playground, API endpoints, student-teacher distillation, data synthesis, load-balancing and orchestration, large context-window mimicry.

Formally, it is a virtual machine/interpreter for human language, coordinating between data, models (CPU), your prompts (code), and tools (IO). 

By doing all these things in one spot in an opinionated way, the LLM-VM can properly optimize batch calls that would be exorbitantly expensive with distributed endpoints.  It furthermore strives for both model and architecture agnosticism, properly optimizing the chosen model for the current architecture.

### 🤌 Why use the Anarchy LLM-VM?

In line with Anarchy's mission, the LLM-VM strives to support open-source models. By utilizing open-source models and running them locally you achieve several benefits:

* **Speed up your AGI development 🚀:** *With AnarchyAI, one interface is all you need to interact with the latest LLMs available.*
  
* **Lower your costs 💸:** *Running models locally can reduce the pay-as-you-go costs of development and testing.*
  
* **Flexibility 🧘‍♀️:** *Anarchy allows you to rapidly switch between popular models so you can pinpoint the exact right tool for your project.*
  
* **Community Vibes 🫂:** *Join our active community of highly motivated developers and engineers working passionately to democratize AGI*
  
* **WYSIWYG 👀:** *Open source means nothing is hidden; we strive for transparency and efficiency so you can focus on building.*

### 🎁 Features and Roadmap

* **Implicit Agents 🔧🕵️:** *The Anarchy LLM-VM can be set up to use external tools through our agents such as **REBEL** just by supplying tool descriptions!*

* **Inference Optimization 🚄:** *The Anarchy LLM-VM is optimized from the agent level all the way to assembly on known LLM architectures to get the most bang for your buck. With state-of-the-art batching, sparse inference and quantization, distillation, and multi-level colocation, we aim to provide the fastest framework available.*

* **Task Auto-Optimization 🚅:** *The Anarchy LLM-VM will analyze your use cases for repetitive tasks where it can activate student-teacher distillation to train a super-efficient small model from a larger more general model without losing accuracy.  It can furthermore take advantage of data-synthesis techniques to improve results.*


* **Library Callable 📚:** *We provide a library that can be used from any Python codebase directly.*

* **HTTP Endpoints 🕸️:** *We provide an HTTP standalone server to handle completion requests.*

* **Live Data Augmentation 📊:** (ROADMAP) *You will be able to provide a live updating data set and the Anarchy LLM-VM will **fine-tune** your models or work with a **vector DB** to provide up-to-date information with citations*

* **Web Playground 🛝:** (ROADMAP) *You will be able to run the Anarchy LLM-VM and test its outputs from the browser.*

* **Load-Balancing and Orchestration ⚖️:** (ROADMAP) *If you have multiple LLMs or providers you'd like to utilize, you will be able to hand them to the Anarchy LLM-VM to automatically figure out which to work with and when to optimize your uptime or your costs*

* **Output Templating 🤵:** (ROADMAP) *You can ensure that the LLM only outputs data in specific formats and fills in variables from a template with either regular expressions, LMQL, or OpenAI's template language*

* **Persistent Stateful Memory 📝:** (ROADMAP) *The Anarchy LLM-VM can remember a user's conversation history and react accordingly*

## 🚀 Quickstart 🚀

### 🥹 Requirements

#### Installation Requirements

Python >=3.10 Supported. Older versions of Python are on a best-effort basis. 

Use ```bash > python3 --version ``` to check what version you are on. 

To upgrade your python, either create a new python env using ```bash > conda create -n myenv python=3.10 ``` or go to https://www.python.org/downloads/ to download the latest version.

     If you plan on running the setup steps below, a proper Python version will be installed for you


#### System Requirements

Different models have different system requirements. Limiting factors on most systems will likely be RAM, but many functions will work at even 16 GB of RAM. 

That said, always lookup the information about the models you're using, they all have different sizes and requirements 
in memory and compute resources. 

### 👨‍💻 Installation

The quickest way to get started is to run `pip install llm-vm` in your Python environment. 

Another way to install the LLM-VM is to clone this repository and install it with pip like so:

```bash
> git clone https://github.com/anarchy-ai/LLM-VM.git
> cd LLM-VM
> ./setup.sh
```

The above bash script `setup.sh` only works for MacOS and Linux.

Alternatively you could do this:

```bash
> git clone https://github.com/anarchy-ai/LLM-VM.git
> cd LLM-VM
> python -m venv <name>
> source <name>/bin/activate
> python -m pip install -e ."[dev]"
```

#### One Last Step, almost there!
If you're using one of the OpenAI models, you will need to set the `LLM_VM_OPENAI_API_KEY` environment
variable with your API key. 


### ✅ Generating Completions
Our LLM-VM gets you working directly with popular LLMs locally in just 3 lines. Once you've installed it (as above), just load your model and start generating!



```python
# import our client
from llm_vm.client import Client

# Select which LLM you want to use, here we have OpenAI 
client = Client(big_model = 'chat_gpt')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '', openai_key = 'ENTER_YOUR_API_KEY')
print(response)
# Anarchy is a political ideology that advocates for the absence of government...
```

### 🏃‍♀ Running LLMs Locally
```python
# import our client
from llm_vm.client import Client

# Select the LlaMA 2 model
client = Client(big_model = 'llama2')

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```


### 😎 Supported Models
Select from the following models
```python

Supported_Models = ['chat_gpt','gpt','neo','llama2','bloom','opt','pythia']
```




### ☯ Picking Different Models
LLM-VM default model sizes for local models are intended to make experimentation 
with LLMs accessible to everyone, but if you have the memory required, larger parameter models 
will perform far better!

for example, if you want to use a large and small neo model  for your teacher and student, and you 
have enough RAM:


```python
# import our client
from llm_vm.client import Client

# Select the LlaMA model
client = Client(big_model = 'neo', big_model_config={'model_uri':'EleutherAI/gpt-neox-20b'}, 
                small_model ='neo', small_model_config={'model_uri':'EleutherAI/gpt-neox-125m'})

# Put in your prompt and go!
response = client.complete(prompt = 'What is Anarchy?', context = '')
print(response)
# Anarchy is a political philosophy that advocates no government...
```
Here are some default model's details:
| Name | Model_Uri | Model params | Checkpoint file size |
|---|---|---|---|
| Neo | `EleutherAI/gpt-neo-1.3B` | 1.3B | 5.31 GB |
| Bloom | `bigscience/bloom-560m` | 1.7B | 1.12 GB |
| OPT | `facebook/opt-350m` | 350m | 622 MB |

For some other choices of memory usage and parameter count in each model family, check out the 
tables [model_uri_tables](./model_uri_tables.md).


### 🛠 Tool Usage

There are two agents: FLAT and REBEL. 

Run the agents separately by going into the `src/llm_vm/agents/<AGENT_FOLDER>` and running the file that is 
titled `agent.py`. 

Alternatively, to run a simple interface and choose an agent to run from the CLI, run the `src/llm_vm/agents/agent_interface.py` file 
and follow the command prompt instructions. 


## 🩷 Contributing 🩷

We welcome contributors!  To get started is to join our active [discord community](https://discord.anarchy.ai).  Otherwise here are some ways to contribute and get paid:

### Jobs

- We're always looking for serious hackers.  Prove that you can build and creatively solve hard problems and reach out! 
- The easiest way to secure a job/internship with us is to submit pull requests that address or resolve open issues.
- Then, you can apply directly here https://forms.gle/bUWDKW3cwZ8n6qsU8

### Bounty

We offer bounties for closing specific tickets! Look at the ticket labels to see how much the bounty is.  To get started, [join the discord and read the guide](https://discord.com/channels/1075227138766147656/1147542772824408074)

## 🙏 Acknowledgements 🙏

- **Matthew Mirman** - CEO
  - GitHub: [@mmirman](https://github.com/mmirman)
  - LinkedIn: [@matthewmirman](https://www.linkedin.com/in/matthewmirman/)
  - Twitter: [@mmirman](https://twitter.com/mmirman)
  - Website: [mirman.com](https://www.mirman.com)

- **Victor Odede** - Undoomer
  - GitHub: [@VictorOdede](https://github.com/VictorOdede)
  - LinkedIn: [@victor-odede](https://www.linkedin.com/in/victor-odede-aaa907114/)

- **Abhigya Sodani** - Research Intern
  - GitHub: [@abhigya-sodani](https://github.com/abhigya-sodani)
  - LinkedIn: [@abhigya-sodani](https://www.linkedin.com/in/abhigya-sodani-405918160/)

- **Carter Schonwald** - Fearless Contributor
  - GitHub: [@cartazio](https://github.com/cartazio)
  - LinkedIn: [@carter-schonwald](https://www.linkedin.com/in/carter-schonwald-aa178132/)
 
- **Kyle Wild** - Fearless Contributor
  - GitHub: [@dorkitude](https://github.com/dorkitude)
  - LinkedIn: [@kylewild](https://www.linkedin.com/in/kylewild/)

- **Aarushi Banerjee** - Fearless Contributor
  - GitHub: [@AB3000](https://github.com/AB3000)
  - LinkedIn: [@ab99](https://www.linkedin.com/in/ab99/)

- **Andrew Nelson** - Fearless Contributor
  - GitHub: [@ajn2004](https://github.com/ajn2004)
  - LinkedIn: [@ajnelsnyc](https://www.linkedin.com/in/ajnelsnyc/)
    
## License

[MIT License](LICENSE)
