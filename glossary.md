# Glossary of Anarchy

## Large Language Models

Large Language Models, often referred to as "LLMs," are a class of artificial intelligence models designed for natural language understanding and generation tasks. These models are characterized by their massive size, extensive training data, and deep neural network architectures. LLMs have gained prominence in recent years due to their ability to generate coherent and contextually relevant text across a wide range of applications.

**Transformer architecture:** a deep learning framework introduced in the paper "Attention Is All You Need." It excels in processing sequential data, such as text, by using a self-attention mechanism to capture dependencies and relationships between elements in a sequence. The architecture employs multi-head attention, allowing it to focus on different aspects of the input simultaneously. To handle the sequence's positional information, positional encodings are added. The Transformer has revolutionized natural language processing and machine translation due to its ability to efficiently model context and dependencies, making it a foundational component of many state-of-the-art language models and NLP applications.

**Attention mechanism:** a fundamental component in deep learning models, particularly in the context of natural language processing (NLP) and computer vision. It enables models to selectively focus on specific parts of input data when making predictions or generating output. This mechanism allows the model to assign varying levels of importance to different elements in a sequence, considering their relevance in the context of the task. By capturing dependencies and context, the attention mechanism has significantly improved the performance of various machine learning applications, including machine translation, text summarization, image captioning, and more.

**Tokenization:** a fundamental natural language processing (NLP) technique that involves breaking down a text into smaller units called "tokens." In the context of language processing, tokens can be words, subwords, or even individual characters, depending on the granularity chosen for analysis. Tokenization is the initial step in many NLP tasks and plays a crucial role in transforming unstructured text data into a format that can be processed by machine learning models. It helps to segment text into meaningful units, making it easier to analyze and manipulate, whether for tasks like text classification, sentiment analysis, machine translation, or language modeling. (one token = 3/4 word or 4 characters)

**Embedding Layer:** a fundamental component of many natural language processing (NLP) and deep learning models, especially those used for text analysis. It is a layer in a neural network that is responsible for converting discrete data, such as words or tokens, into fixed-size continuous vector representations, also known as word embeddings or token embeddings. Words with similar meanings or related meanings are represented close to each other in the vector space, allowing the model to capture semantic relationships.

**Prompt:** refers to the input or query provided to the model to elicit a response or generate text. The prompt is a text-based instruction or question that serves as a starting point for the LLM to generate a coherent and contextually relevant response.

**Transfer learning** is a machine learning technique that reuses a pretrained model's knowledge on one task to enhance performance on a related task. It's a time-efficient way to improve models, particularly when data for the target task is limited.

**Knowledge distillation:** a process where a smaller or more efficient model (the "student") is trained to replicate the behavior and predictions of a larger, more complex model (the "teacher"). Advantage: more computationally efficient and suitable for deployment in resource-constrained environments.

**Quantization**: a process of reducing the precision or bit-width of numerical values, typically weights and activations, used in a neural network model. This involves representing the original high-precision floating-point values with a limited set of discrete values, often integers.
It makes neural network models more memory-efficient and computationally faster during inference, without significantly compromising their performance.Quantization is particularly valuable for deploying models on resource-constrained devices like mobile phones, IoT devices, and edge computing devices.

**Distributed learning**, also known as distributed machine learning, refers to the practice of training machine learning models across multiple computing devices or nodes in a distributed computing environment. The goal of distributed learning is to leverage the computational power and resources of multiple machines to accelerate the training process and handle larger datasets and more complex models.

**Pruning:** is a technique used to remove certain connections (weights) or neurons from a neural network while retaining its general structure and functionality. These connections or neurons are often identified based on their low importance or contribution to the network's overall performance.

- **Purpose:** The main purpose of pruning is to reduce the size of a neural network model, thereby decreasing memory usage and computational requirements during inference. Smaller models are more efficient and suitable for deployment on resource-constrained devices.

- **Methods:** Various methods and criteria can be used for pruning, including magnitude-based pruning (removing small-weight connections), sensitivity analysis, and iterative pruning. Pruning can be applied to different layers or parts of a neural network, depending on the specific goals of model compression and optimization.

**Sparsity:** refers to the property of having many of the elements or parameters in a neural network set to zero. A sparse neural network contains a significant portion of zero-valued parameters, resulting in a more compact representation.

- **Purpose:** Introducing sparsity into a neural network is often an outcome of pruning, but it can also be achieved through other techniques. Sparse models consume less memory and require fewer computations, making them suitable for deployment in scenarios where computational resources are limited.

- **Benefits:** Sparse neural networks can have improved inference speed, reduced memory footprint, and decreased energy consumption, which is advantageous for edge devices, mobile applications, and real-time systems.

- **Sparsity Techniques:** Besides pruning, techniques like weight regularization with sparsity-inducing penalties (e.g., L1 regularization), structured sparsity, and quantization can be used to induce sparsity in neural network models.

### Hallucination

**Hallucination:** A phenomenon where the model generates text or responses that contain information or details that are not based on factual data or reality. It occurs when the model produces information that is fabricated, imagined, or fictional rather than being grounded in accurate information from its training data or the real world.

**Causes of Hallucination:**

    1. Biases in the training data.
    2. Limitations in the model's understanding of context.
    3. The model's tendency to generate creative but incorrect information.

Researchers and developers continually work to mitigate and reduce hallucination in LLMs to make them more reliable and trustworthy in their outputs.
