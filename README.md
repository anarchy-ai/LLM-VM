# LLM-VM

This is an LLM agnostic JIT for natural language. Specifically, it uses LLMs to convert conversational natural language into a dynamic series of LLM and IO commands. You provide the underlying provider(s), actions (APIs, code-hooks) and their descriptions, data-sources (PDFs, websites...), and the LLM-VM will take care of load-balancing, fine-tuning, natural language compilation and tool-selection.

This is still in BETA.  Very little attention has been paid to package structure.  Expect it to change.

## Running/Testing

The following instructions assume you have a venv running. See <https://docs.python.org/3/library/venv.html> for help.

```bash
echo "OPENAI_KEY=<YOUROPENAIKEY>" >> .env
pip3 install -r requirements.txt
python3 src/llm_vm/completion/test_optimize.py
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
