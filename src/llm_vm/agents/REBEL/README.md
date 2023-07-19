![Anarchy Logo](../../../../anarchy_logo.svg)
**Abstract**

While large language models (LLMs) have demonstrated impressive performance in question answering tasks, their performance is limited when the questions require knowledge that is not
included in the model’s training data and can only be acquired through direct observation or interaction with the real world. Existing methods decompose reasoning tasks through the use of modules invoked sequentially, limiting their ability to answer deep reasoning tasks. We introduce a method, Recursion based extensible LLM
(REBEL), which handles open-world, deep reasoning tasks by employing automated reasoning techniques like dynamic planning and forward chaining strategies. REBEL allows LLMs to reason via recursive problem decomposition and utilization of external tools. The tools that REBEL
uses are specified only by natural language description. We further demonstrate REBEL capabilities on a set of problems that require a deeply nested use of external tools in a compositional
and conversational setting.

**Evaluation**

In this section we first introduce the experimental setup, including the benchmarks used for evaluation, and then present the results.

**Experimental Setup**

We tested REBEL on 3 datasets: Compositional Celebrities (Ofir Press, 2022), FEVER (Thorne et al., 2018), and
HotPotQA (Yang et al., 2018). On these datasets, correctness was determined by a human
experimenter based on the output of each system. ReAct outputs with simply the answer to the question, while REBEL
often outputs the answer wrapped in reasoning behind the system’s thoughts. For these experiments, two separate sets
of rules had to be determined for fact verification and fact retrieving questions. For fact retrieving questions, an answer
was considered correct if the desired answer was contained in the system output. For fact verification, if the model output determination of the truthfulness of a statement was the
same as the desired truthfulness, then the generated answer was considered correct. On Compositional Celebrities, due to computational limitations, we tested using 5 of the 17 categories available,
using 100 questions per category, randomly chosen. We tested on FEVER and HotPotQA with 100 of the same random questions from each dataset on both ReAct and
REBEL. FEVER has 3 types of potential output labels (SUPPORTS, REFUTES, NOT ENOUGH INFO). In order to make prevent accidental correct answers from the
REBEL system, only questions with the SUPPORTS and REFUTES labels were considered. For this experiment REBEL was only allowed to use a search tool to query the internet, as that is the only tool
that the ReAct system has access to. Our code, which can be found at rebel.anarchy.ai, was implemented in Python using the OpenAI Completion API to access GPT-3 (da-vinci-003).

**Results**

We found that REBEL outperformed ReAct on answering questions that require i) the gathering of many facts to determine an answer ii) very specific search queries that return
large amounts of unstructured data. With our experimental results we were able to show that REBEL is a state-of-the-art system in terms of its ability to consistently answer
questions from disparate knowledge bases.

Below is the table depicting the results of the REBEL system versus ReAct on Compositional Celebrities.

<img width="400" alt="image" src="https://github.com/anarchy-ai/LLM-VM/assets/37461794/842ff756-f52b-403d-94a0-595e5ac9bba7">

Below is the table depicting the results of the REBEL systhem versus ReAct on HotPotQA and FEVER.

<img width="300" alt="image" src="https://github.com/anarchy-ai/LLM-VM/assets/37461794/de2e1df4-7f4a-4947-8c32-2ce4312df484">


