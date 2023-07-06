# LLM-VM

This is an LLM agnostic JIT for natural language. Specifically, it uses LLMs to convert conversational natural language into a dynamic series of LLM and IO commands. You provide the underlying provider(s), actions (APIs, code-hooks) and their descriptions, data-sources (PDFs, websites...), and the LLM-VM will take care of load-balancing, fine-tuning, natural language compilation and tool-selection.

This is still in BETA.  Very little attention has been paid to package structure.  Expect it to change.

## Running/Testing

To run and test this repository you need to start a flask server. To start the flask server use: 

```bash
pip3 install -r requirements.txt
python3 app.py
```

This will start a flask server as http://192.168.1.75:3002 and will create an endpoint http://192.168.1.75:3002/completion. Post requests can be sent to this endpoint in the following format:

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
If no temperature, finetune, or data_synthesis are provided, these values will default to 0, false, and false. 

To access the completion endpoint programmatically, after starting up the server on your machine, import llm_vm.py into your code and call the completion function. To test that the server is working correctly, after starting it, run 
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

## License

`llm_vm` was created by Matthew Mirman. It is licensed under the terms of the MIT license.
