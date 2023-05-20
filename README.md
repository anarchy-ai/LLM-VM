# LLM-VM

This is an LLM agnostic JIT for natural language. Specifically, it uses LLMs to convert conversational natural language into a dynamic series of LLM and IO commands. You provide the underlying provider(s), actions (APIs, code-hooks) and their descriptions, data-sources (PDFs, websites...), and the LLM-VM will take care of load-balancing, fine-tuning, natural language compilation and tool-selection.

This is still in BETA.  Very little attention has been paid to package structure.  Expect it to change.

## Running/Testing

```bash
$ echo "OPENAI_KEY=<YOUROPENAIKEY>" >> .env
$ pip3 --install requirements.txt
$ python3 src/llm_vm/completion/test_optimize.py
```

## License

`llm_vm` was created by Matthew Mirman. It is licensed under the terms of the MIT license.
