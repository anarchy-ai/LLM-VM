# LLM-VM

This is an LLM agnostic JIT for natural language. Specifically, it uses LLMs to convert conversational natural language into a dynamic series of LLM and IO commands. You provide the underlying provider(s), actions (APIs, code-hooks) and their descriptions, data-sources (PDFs, websites...), and the LLM-VM will take care of load-balancing, fine-tuning, natural language compilation and tool-selection.
