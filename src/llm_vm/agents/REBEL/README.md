![Anarchy Logo](../../../../anarchy_logo.svg)

To run the REBEL agent run 
```
python quickstart_REBEL.py
```
in the root LLM-VM directory. In the quickstart_REBEL.py file it is shown how one can use the LLM-VM client function to declare tools and use the REBEL agent. 

Below we present data on the REBEL agent and its merits. 

📚**Abstract**📚

* While large language models (LLMs) have demonstrated impressive performance in question answering tasks, their performance is limited when the questions require knowledge that is not included in the model’s training data and can only be acquired through direct observation or interaction with the real world. 🔍
* Existing methods decompose reasoning tasks through the use of modules invoked sequentially, limiting their ability to answer deep reasoning tasks. 🧠
* We introduce a method, Recursion based extensible LLM (REBEL), which handles open-world, deep reasoning tasks. REBEL allows LLMs to reason via recursive problem decomposition and utilization of external tools. 🚀

📚**Evaluation**📚

In this section we first introduce the experimental setup, including the benchmarks used for evaluation, and then present the results. 🤔

📚**Experimental Setup**📚

* We tested REBEL on 3 datasets: Compositional Celebrities (Ofir Press, 2022), FEVER (Thorne et al., 2018), and
HotPotQA (Yang et al., 2018).
* On these datasets, correctness was determined by a human experimenter based on the output of each system. ReAct outputs with simply the answer to the question, while REBEL
often outputs the answer wrapped in reasoning behind the system’s thoughts.
* Our code, which can be found at in this directory, was implemented in Python using the OpenAI Completion API to access GPT-3 (da-vinci-003).

📚**Results**📚

*We found that REBEL outperformed ReAct on answering questions that require i) the gathering of many facts to determine an answer ii) very specific search queries that return large amounts of unstructured data. 

Below is the table depicting the results of the REBEL system versus ReAct on Compositional Celebrities.

<img width="400" alt="image" src="https://github.com/anarchy-ai/LLM-VM/assets/37461794/842ff756-f52b-403d-94a0-595e5ac9bba7">

Below is the table depicting the results of the REBEL systhem versus ReAct on HotPotQA and FEVER.

<img width="300" alt="image" src="https://github.com/anarchy-ai/LLM-VM/assets/37461794/de2e1df4-7f4a-4947-8c32-2ce4312df484">


