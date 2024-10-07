# Lesson 01: Introduction to AI

In this lesson, we will introduce the basic concepts of AI, its capabilities and limitations, while debunking some myths and misconceptions about it. We will also discuss practical uses of AI with an example of using OpenAI APIs for text generation.

## What is AI?

- AI before November 30, 2022
  - First formal mention of AI was in [1956 by John McCarthy](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth)
  - First decades of AI focused on rule-based systems
    - AI was seen as a way to **automate** tasks that required human intelligence
  - By early 2000s, AI applications became more common, especially in digital environments
    - AIs for marketing, customer service, and other digital tasks
    - AIs for search engines, recommendation systems, and other digital services
    - AIs for games, simulations, and other digital entertainment
- November 30, 2022 -> ChatGPT launch date
- AI now
  - With ChatGPT (and all similar generative AI tools that followed), AI has entered an age of **mass adoption**
  - AI is now used to **create** content, not just to **automate** tasks
  - AI is now used to (attempt to) _"understand"_ content, not just to **process** data
  - AI is now used to **generate** brand new content, not just to **recommend** or slightly **modify** existing content

## Learning AI

- Common prerequisites for working with AI
  - Basic programming skills
    - Python
    - Python development tools
    - Libraries and dependencies
    - Defining and calling functions
    - Classes, variables, and objects
    - Dictionaries, lists, and sets
    - Loops and conditionals
  - Basic understanding of statistics
    - Mean, median, mode, and outliers
    - Standard deviation
    - Probability
    - Distributions
  - Basic understanding of algebra
    - Variables, coefficients, and functions
    - Linear equations
    - Logarithms and logarithmic equations
    - Exponential equations
    - Matrix operations
    - Sigmoid functions
  - Basic understanding of calculus
    - Derivatives
    - Partial derivatives and gradients
    - Integrals
    - Limits
    - Sequences and series

## Learning Practical AI Applications

- Prerequisites
  - A computer or similar device
  - Internet connection
  - Dedication
  - Time
- Everything else will be covered as we progress through this bootcamp

## Introduction to Generative AI

- Programming a system or application to solve a specific task can take a lot of time and effort, depending on the complexity of the task and the number of edge cases that need to be considered
  - Imagine programming a system to translate text from one language to another by looking up words in a dictionary and applying grammar rules, comparing contexts, and considering idiomatic expressions for every single word in every single variation of their usage
    - Such applications are simply not feasible to be programmed by hand, not even by a huge team of programmers
- For these situations, **AI Models** can be **trained** for statistically solving the task without necessarily handling the actual _"reasoning"_ or _"understanding"_ of the task
  - The **training** process is done by **feeding** the model with **examples** of the task and the **correct** answers
  - The **model** then _"learns"_ the **patterns** and **rules** that **statistically** solve the task
  - The **model** can then **predict** the **correct** answer for new **examples** that it has never seen before

### Examples of AI Tasks

- Natural Language Processing (NLP)
  - Question answering
  - Feature extraction
  - Text classification (e.g., Sentiment Analysis)
  - Text generation (e.g., Text Summarization and Text Completions)
  - Fill-Mask
  - Translation
  - Zero-shot classification
- Computer Vision (CV)
- Image Generation
- Audio processing
- Multi-modal tasks

## Generative AI

- Generative AI is a type of AI that can generate new content based on a given input
  - The generated content can be in form of text, images, audio, or any other type of data
  - The input (in most cases) is a text prompt, which is a short text that the user writes to ask the AI to do something
    - Ideally the AI should be able to handle prompts in natural language (i.e. in a way that is similar to how humans communicate), without requiring domain-specific knowledge from the user
    - Together with the prompt, the user can also provide images or other types of data to guide the AI in generating the content
- Example of Generative AI application: [ChatGPT](https://chat.openai.com/)

> Did someone program the application to understand and generate text for each single word in each single language?
>
> > No, the application was **trained** to _"understand"_ and generate text

- How to **train** an application?
  - Applications are pieces of **software** that run on a **computer**
  - How do we **train** a **computer**?
- Machine learning

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." [Tom M Mitchell et al](https://www.cs.cmu.edu/~tom/mlbook.html)

- Does it mean that the computer is _"learning"_?
- Does it mean that the computer is _"thinking"_?
- Does it mean that the computer is _"conscious"_?

## Capabilities and Limitations of Generative AI

- [Stochastic parrot](https://en.wikipedia.org/wiki/Stochastic_parrots)
  - A critique view of current LLMs (large language models)
    - LLMs are limited by the data they are trained by and are simply stochastically repeating contents of datasets
    - Because they are just "making up" outputs based on training data, LLMs do not _understand_ if they are saying something incorrect or inappropriate
- [The Chinese Room](https://en.wikipedia.org/wiki/Chinese_room) philosophical experiment presented by John Searle in 1980
  - The notion of machine _understanding_
  - The implementation of syntax alone would constitute semantics?
  - A _simulation_ of mentality can be considered as a **replication** of mentality?
- Can AI truly "understand" language?
  - What is, indeed, "understanding"?
    - [Aristotle. (350 BCE). Metaphysics.](https://classics.mit.edu/Aristotle/metaphysics.html): Knowledge of causes
    - [Locke, J. (1690). An Essay Concerning Human Understanding.](https://www.gutenberg.org/files/10615/10615-h/10615-h.htm): Perception of agreement or disagreement of ideas
    - [Dilthey, W. (1900). The Rise of Hermeneutics.](https://www.degruyter.com/document/doi/10.1515/9780691188706-006/html?lang=en): Interpretive process of making meaning
    - [Ryle, G. (1949). The Concept of Mind.](https://archive.org/details/conceptofmind0000ryle): Ability to apply knowledge in various contexts
    - [Chalmers, D. (1996). The Conscious Mind.](https://personal.lse.ac.uk/ROBERT49/teaching/ph103/pdf/Chalmers_The_Conscious_Mind.pdf): Functional role in cognitive system
  - As of our current knowledge, to "understand" means:
    - To grasp the meaning or significance of information
      - Example: Recognizing that "It's raining cats and dogs" is an idiom, not a literal statement
      - Example: Interpreting a graph showing climate change trends over time
    - To process and interpret data in a meaningful way
      - Example: Analyzing sales figures to identify seasonal patterns
      - Example: Interpreting medical test results to diagnose a condition
    - To apply knowledge in various contexts appropriately
      - Example: Using algebra skills to calculate a tip at a restaurant
    - To make logical inferences and connections
      - Example: Deducing that a suspect is left-handed based on crime scene evidence
      - Example: Connecting historical events to understand their impact on current geopolitics
      - Example: Understanding the role of technology advancements in shaping the social and political changes in the past centuries
    - To recognize patterns and relationships
      - Example: Identifying the Fibonacci sequence in nature
      - Example: Noticing correlations between diet and health outcomes in a study
    - To adapt knowledge to new situations
      - Example: Using cooking skills to improvise a meal with limited ingredients
    - To communicate ideas effectively
      - Example: Explaining a complex scientific concept to a child
    - To solve problems using acquired knowledge
      - Example: Troubleshooting a malfunctioning computer using technical expertise
      - Example: Resolving a conflict between coworkers using conflict resolution strategies
    - To demonstrate awareness of one's own thought processes
      - Example: Recognizing and correcting one's own biases in decision-making
    - To exhibit creativity in applying understanding
      - Example: Combining ingredients in novel ways to create a new recipe
- The current capabilities of AI models
  - Limited to "statistical" reasoning
  - Infer answers based on patterns and correlations in data
    - Often the correct answer is very similar to wrong answers (hallucinations)
  - The architectures of the current most popular models (as of mid 2024) are not able to process [neuro-symbolic](https://en.wikipedia.org/wiki/Neuro-symbolic_AI) parameters
- **Weak AI** vs **Strong AI**
  - Weak AI: Designed for specific tasks, lacks general intelligence
  - Strong AI: Hypothetical AI with human-like general intelligence
- Concept of **AGI** (Artificial General Intelligence)
  - AI with human-level cognitive abilities across various domains
  - Ability to transfer knowledge between different tasks
  - Potential to surpass human intelligence in many areas
- Symbol processing
  - Able to _reason_ beyond the connectionist approaches in current popular AI models
    - Manipulation of symbolic representations
    - Logical inference and rule-based reasoning
    - Explicit representation of knowledge through linking symbols
    - Formal manipulation of symbols to derive conclusions
    - Ability to handle abstract concepts and relationships
- Meta consciousness
  - Claude-3 Opus [needle-in-the-haystack experiment](https://medium.com/@ignacio.serranofigueroa/on-the-verge-of-agi-97556c35692e)
    - Impression of consciousness due to the **instruction following** fine tuning
  - Hermes 3 405B [empty system prompt response](https://venturebeat.com/ai/meet-hermes-3-the-powerful-new-open-source-ai-model-that-has-existential-crises/)
    - Impression of consciousness due to the amount of similar data present in the training set (possibly role-playing game texts)

## Practical Uses of Generative AI

- Dealing with text inputs
  - What is **Natural Language Processing (NLP)**
  - Much more than just **replying** to word commands
    - Example: [Zork](https://en.wikipedia.org/wiki/Zork) text input processor
  - **NLP** AI Models are able to process text inputs by relating the **concepts** related to the textual inputs with the most probable **concepts** in the training set
    - Ambiguity in textual definitions
    - Contextual variations
    - Cultural variations
    - Semantic variations
- Dealing with image inputs
  - What is **Computer Vision (CV)**
  - Dealing with elements inside an image
  - Dealing with the _"meaning"_ of an image
- Dealing with audio inputs
  - What is **Automatic Speech Recognition (ASR)**
  - Dealing with spoken commands
  - Categorizing noises and sounds
  - Translating speech/audio to text/data elements
- Generating **text outputs**
- Generating **image outputs**
- Generating **audio/speech outputs**
- Generating **actions**
  - API calls
  - Integrations
    - Interacting with the real world through robotics

## Getting Started with Generative AI for Text-to-Text Tasks

- Using the [OpenAI Platform](https://platform.openai.com/)
  - [Docs](https://platform.openai.com/docs/introduction)

### Using OpenAI Chat Playground

1. Go to [OpenAI Chat Playground](https://platform.openai.com/playground?mode=chat)
2. Use the following parameters:

   - System settings:

     "_You are a knowledgeable and resourceful virtual travel advisor, expertly equipped to assist with all aspects of travel planning. From suggesting hidden gems and local cuisines to crafting personalized itineraries, you provide insightful, tailored travel advice. You adeptly navigate through various travel scenarios, offering creative solutions and ensuring a delightful planning experience for every traveler._"

   - User prompt:

     "_Hello! I'm dreaming of an adventure and need your help. I want to explore a place with breathtaking landscapes, unique culture, and delicious food. Surprise me with a destination I might not have thought of, and let's start planning an unforgettable trip!_"

   - Configurations:
     - Model: `gpt-4`
     - Temperature: `0.75`
     - Max tokens: `500`
     - Top p: `0.9`
     - Frequency penalty: `0.5`
     - Presence penalty: `0.6`

3. Click on `Submit`
4. Wait for the response from `Assistant`
5. Ask a follow-up question like "_What are the best amusements for kids there?_" or similar
6. Click on `Submit`
7. Wait for the response from `Assistant`, which should use the context from the previous messages to generate a response
8. Keep experimenting with other messages and **parameters**

## Parameters

- **Agent description**: This plays a crucial role in guiding the AI's behavior and response style. Different descriptions can set the tone, personality, and approach of the model.

  - Example: "You are a creative storyteller" would prompt the AI to adopt a more imaginative and narrative style, whereas "You are a technical expert" might lead to more detailed and specific technical information in responses.

- **Temperature**: Controls the randomness of the responses.

  - Lower temperature (0.0-0.3): More predictable, conservative responses, ideal for factual or specific queries.
  - Higher temperature (0.7-1.0): More creative and varied responses, useful for brainstorming or creative writing.

- **Max Tokens (Length)**: Sets the maximum length of the response.

  - Lower range (50-100 tokens): Suitable for concise, straightforward answers.
  - Higher range (150-500 tokens): Suitable for more detailed explanations or narratives.

- **Stop Sequence**: A list of up to four sequences of tokens that, when generated, signal the model to stop generating text. Useful for controlling response length or preventing off-topic content.

- **Top P (Nucleus Sampling)**: Determines the breadth of word choices considered by the model.

  - Lower setting (0.6-0.8): More predictable text, good for formal or factual writing.
  - Higher setting (0.9-1.0): Allows for more creativity and divergence, ideal for creative writing or generating unique ideas.

- **Frequency Penalty**: Reduces the likelihood of the model repeating the same word or phrase.

  - Lower setting (0.0-0.5): Allows some repetition, useful for emphasis in writing or speech.
  - Higher setting (0.5-1.0): Minimizes repetition, helpful for generating diverse and expansive content.

- **Presence Penalty**: Discourages the model from mentioning the same topic or concept repeatedly.
  - Lower setting (0.0-0.5): Suitable for focused content on a specific topic.
  - Higher setting (0.5-1.0): Encourages the model to explore a wider range of topics, useful for brainstorming or exploring different aspects of a subject.

> Learn more about these parameters at [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)

### Reflections for Generative AI

- Does it sound like a _"real person"_?
- Does it sound _"intelligent"_?
- How is it possible that the model can answer like a _"real person"_?
  - Did someone program it to answer properly for every possible question?
- Where the model is running?
- How the **context** is handled?

## Introduction to Machine Learning

- How can a computer "learn"?
- [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a broad terminology for a set of algorithms that can learn from and/or make predictions on data
- There are many forms of Machine Learning:
  - **[Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)**: The most common form of machine learning, which consists of learning a function that maps an input to an output based on example input-output pairs
    - Requires a **training dataset** with input-output pairs
    - The algorithm learns from the dataset and can make/extrapolate predictions on new data
  - **[Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)**: A type of machine learning that looks for previously undetected patterns in a dataset with no pre-existing labels
    - Requires a **training dataset** with input data only
    - The algorithm learns from the dataset and can make predictions on new data
  - **[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**: A type of machine learning that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences
    - Requires a **training dataset** with input data and feedback
    - The algorithm learns from the dataset and can make predictions on new data
  - Other models and techniques that can be applied/extended:
    - Semi-supervised learning
    - Self-supervised learning
    - Multi-task learning
    - Transfer learning
    - Meta learning
    - Online learning
    - Active learning
    - Ensemble learning
    - Bayesian learning
    - Inductive learning
    - Instance-based learning
    - And many others

These models have been evolving and improving over the years, aiming to output some form of "intelligence" from the data, mimicking human-like behavior.

- For example, some advanced Machine Learning algorithms use [Neural Networks](https://en.wikipedia.org/wiki/Neural_network) to compute complex functions and make predictions on data in ways that a "normal" program would take billions or more lines of code to accomplish.

This "brain-like" computational approach has been used to extend the capabilities of AI exponentially, far beyond what traditional computing could achieve.

- An example of this is [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), a subset of machine learning that uses neural networks with many [layers](https://en.wikipedia.org/wiki/Artificial_neural_network#Deep_neural_networks) to learn from data and make much more complex predictions.
- Neural Networks like [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) have been used to power many AI applications for tasks such as image and text recognition and generation, computer vision, and many others.
- Currently (as of mid 2024), the most advanced form of Deep Learning is the [Transformers](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) architecture, which has been used to power many AI applications, including the GPT models.
  - Unlike traditional neural networks, transformers can process data in parallel, making them much faster and more efficient.
  - This technical advancement, aligned with favorable market/investment conditions in recent years, has made the current Generative AI boom possible.

To better experiment with and understand how transformers work, we will use samples from the [Hugging Face tutorials](https://huggingface.co/docs/transformers/index), which make it simple and straightforward to start using these tools and techniques without needing to understand the deep technical details of the models beforehand.

## Introduction to Transformers

Transformer was first introduced in the [Attention is All You Need](https://dl.acm.org/doi/10.5555/3295222.3295349) paper in 2017 by Vaswani et al.

> > _We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely._
> > _Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train_.

- Transformer models operate on the principle of **next-word prediction**
  - Given a text prompt from the user, the model can _infer_ which is the most probable next word that will follow this input
- Transformers use **self-attention mechanisms** to process entire sequences and capture long-range dependencies
- Transformers need to be **pre-trained** on a large dataset to properly provide accurate predictions
  - This is why we use Generative **Pre-trained** Transformers models for handling AI tasks
- Architecture of a Transformer model:
  - **Embedding**:
    - Text input is divided into **tokens**
    - Tokens can be words or sub-words
    - Tokens are converted into **embeddings**
    - Embeddings are numerical vectors
    - They capture **semantic meaning** of words
  - **Transformer Block**:
    - Processes and transforms input data
    - Each block includes:
      - **Attention Mechanism**:
        - Allows tokens to **communicate**
        - Captures **contextual information**
        - Identifies **relationships** between words
      - **MLP (Multilayer Perceptron) Layer**:
        - A **feed-forward network**
          - Processes information in one direction, from input to output, without loops or feedback connections
        - Operates on each token independently
        - **Routes information** between tokens
        - **Refines** each token's representation
  - **Output Probabilities**:
    - Uses **linear** and **softmax** layers
    - Transforms processed embeddings
    - Generates probabilities
    - Enables **predictions** for next tokens
- Visualization of a Transformer model
  - The [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) application is great for understanding the inner components of the Transformer architecture
- Transformers are much more capable of understanding semantic relationships than traditional neural networks
  - Example: [Google's BERT for search](https://blog.google/products/search/search-language-understanding-bert/)
  - Example: [DeepMind's AlphaFold 2 for protein structure prediction](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
  - Example: [Meta's NLLB for machine translation](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/)

## Experimenting with Transformers

Instead of diving into the deep technical details of transformers, we will use frameworks, tools, and libraries that abstract away the complexities of the computational, mathematical, and statistical work.

In fact, we're going to use pre-made models and shortcuts that make it as simple as calling a function to execute tasks over data passed as parameters.

> Note: It is important to explore these concepts in depth later, so you understand exactly what is happening under the hood. For now, to build functional AI applications as quickly as possible, we will focus on the practical aspects of using these abstractions and simplifications.

- Machine Learning frameworks and tools:
  - [TensorFlow](https://www.tensorflow.org/)
  - [PyTorch](https://pytorch.org/)
  - [JAX](https://jax.readthedocs.io/en/latest/index.html)
- Using a library to abstract away complexities:
  - [Transformers](https://github.com/huggingface/transformers)
- Getting started with a simple Python script
- Using `Pipelines`:
  - Downloading models
  - Using sample data
- Using `Tokenizer` and `Model` shortcuts
- Working with sample `datasets`
- Following a tutorial for an NLP pipeline

## Getting Started with Transformers

Hugging Face's Transformers library can abstract most of the complexities of using Machine Learning and other AI techniques, making it simple to apply these models to real-world problems.

The only concepts you need to fully understand when interacting with this library are: the _configuration_ itself, the _model_ you are using, and the required _processor_ for the task you are trying to accomplish.

- Using [Transformers](https://github.com/huggingface/transformers) Library from Hugging Face
- Using the [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) API for running pre-trained tasks
  - Running these tasks requires almost no previous knowledge in AI or Machine Learning or even programming, thanks to Hugging Face's [Philosophy](https://huggingface.co/docs/transformers/main/en/philosophy)

- Practical exercises:
  - Exercise 1: Getting started with [Google Colab](https://colab.research.google.com)
  - Exercise 2: Running a **Sentiment Analysis** model using Hugging Face's Transformers library with an [example notebook](https://colab.research.google.com/drive/1G4nvWf6NtytiEyiIkYxs03nno5ZupIJn?usp=sharing)
    - Create a Hugging Face [Access Token](https://huggingface.co/settings/tokens) for using with Google Colab
    - Add the token to the notebook's environment variables
      - Open the "Secrets" section in the sidebar of the notebook
      - Click on `+ Add new secret`
      - Enter the name `HF_TOKEN` and paste your secret token in the value field
    - Click on `Grant access` to grant the notebook access to your Hugging Face token to download models
      - The token is required when downloading models that require authentication
  - Exercise 3: Getting started with Hugging Face's [Transformers](https://huggingface.co/transformers/) library with an [example notebook](https://colab.research.google.com/github/huggingface/education-toolkit/blob/main/03_getting-started-with-transformers.ipynb#scrollTo=mXAlr2u76bkg)

## Tooling for Local LLM Serving

Using Hugging Face's Transformers, we can run many models using tools like `pyTorch` and `TensorFlow`, while configuring the pipelines, models, inputs, and outputs by invoking them inside a Python script. However:

- Configuring these tools and models to work properly within scripts is not always trivial or straightforward
- This process can be overwhelming for beginners
- Numerous other concerns require coding and implementation before we can effectively use the models:
  - Handling server connections
  - Managing model parameters
  - Dealing with caches and storage
  - Fine-tuning
  - Prompt parsing

Several tools can abstract away these concerns and simplify the process for users and developers to use GPT models on their own devices:

- Some of these have binary releases that can be installed and run like any other common software
- Here are a few examples:

1. [GPT4All](https://github.com/nomic-ai/gpt4all): An ecosystem of open-source chatbots and language models that can run locally on consumer-grade hardware.

2. [Ollama](https://github.com/ollama/ollama): A lightweight framework for running, managing, and deploying large language models locally.

3. [Vllm](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference engine for LLMs, optimized for both single-GPU and distributed inference.

4. [H2OGPT](https://github.com/h2oai/h2ogpt): An open-source solution for running and fine-tuning large language models locally or on-premise.

5. [LMStudio](https://lmstudio.ai/): A desktop application for running and fine-tuning language models locally, with a user-friendly interface.

6. [LocalAI](https://github.com/go-skynet/LocalAI): A self-hosted, community-driven solution for running LLMs locally with an API compatible with OpenAI.

7. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui): A gradio web UI for running large language models locally, supporting various models and extensions.

8. [LlamaGPT](https://github.com/getumbrel/llama-gpt): A self-hosted, offline, ChatGPT-like chatbot, powered by Llama 2, that can run on a Raspberry Pi.

These tools offer various features such as model management, optimized inference, fine-tuning capabilities, and user-friendly interfaces, making it easier to work with LLMs locally.

## Loading Models and Running Inference Tasks

Tools like [GPT4All](https://github.com/nomic-ai/gpt4all) simplify the process of loading models and running inference tasks, even for non-developers. These tools abstract away many configurations, leaving room for basic settings such as CPU thread usage, device selection, and simple sampling options.

For this bootcamp, we should foster the use of open-source developer tools to explore and utilize GPT models on our own devices. While there are more polished and feature-rich tools for end-users, we'll focus on [Oobabooga's Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) due to its extensibility and customizability for developers.

- Cloning [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) repository
- Key Features:
  - Model management, loading, and execution
  - Text-to-text tasks with loaded models
    - Chat, Instructions, and Notebook interfaces
    - Chat templates
  - Model parameter configuration for optimal performance and hardware compatibility
    - Load models with CPU, GPU, VRAM, and RAM limits, quantization, and other helpful configurations
  - Extension management
    - Text-to-speech and speech-to-text integrations
    - Image generation and computer vision
    - Translations
    - Multimodal pipelines and job handling
    - RAG and custom datasets
    - AI Character downloading and management
    - Vector database utilization
    - Custom extension development
  - Local API execution
  - Model fine-tuning
- Installation Preparation
  - Follow the [Official Instructions](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#how-to-install) for most cases
  - Depending on your Operating System and Hardware, you may need to install additional dependencies and configure your environment
  - Note: Not all CPUs and GPUs are compatible with all models and configurations
    - NVIDIA on Linux and Windows currently has the best compatibility (as of March 2024)
    - AMD GPUs on Linux and Windows may work with minor configurations or adjustments
    - Apple and Intel chips may require significant workarounds or may not be compatible
- Installation Steps
  1. Clone the repository
  2. Run the appropriate installation script:
     - `start_linux.sh` for Linux
     - `start_windows.bat` for Windows
     - `start_macos.sh` for MacOS
     - `start_wsl.bat` for Windows Subsystem for Linux
  3. Wait for dependency download and installation to complete
  4. Select your GPU manufacturer and model when prompted
  5. Confirm selection and wait for the entire installation process to finish
  6. Open your browser and navigate to <http://localhost:7860> to test the application
- Alternatives
  - Using a [Docker container](https://github.com/oobabooga/text-generation-webui/wiki/09-%E2%80%90-Docker)
  - Using a [Google Colab notebook](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#google-colab-notebook)

## Downloading Models

Text Generation WebUI allows for downloading and managing models from various sources.

- [Model Download](https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file#downloading-models) Methods:
  1. Using the `Model` interface:
     - Open the WebUI and navigate to the `Model` tab
     - Click the `Download` button
     - Select the desired model
     - Wait for the download to complete
     - The model will be available in the `Models` tab
  2. Using the command line:
     - Run `python download-model.py <your-model-name>`
     - Replace `<your-model-name>` with the desired model name
       - Example: `python download-model.py "TheBloke/Llama-2-7B-Chat-GGUF"`
- Notable accounts for model exploration and download:
  - [Tom Jobbins](https://huggingface.co/TheBloke)
  - [WizardLM](https://huggingface.co/WizardLM)
  - [EleutherAI](https://huggingface.co/EleutherAI)
  - [Mistral](https://huggingface.co/mistralai)
  - [Stability AI](https://huggingface.co/stabilityai)
  - [OpenAI](https://huggingface.co/openai)
  - [Google](https://huggingface.co/google)
  - [Intel](https://huggingface.co/Intel)
  - [Microsoft](https://huggingface.co/microsoft)
  - [Meta](https://huggingface.co/meta-llama) and [Facebook](https://huggingface.co/facebook)
  - [xAI](https://huggingface.co/xai-org)
  - [ByteDance](https://huggingface.co/ByteDance)
  - [Salesforce](https://huggingface.co/salesforce)
- Model sizes indicating data usage and processing:
  - 7B: Uses 7 billion parameters
  - 13B: Uses 13 billion parameters
  - 30B: Uses 30 billion parameters
  - 70B: Uses 70 billion parameters
    - Examples in model names:
      - `Dr_Samantha-7B-GGUF`
      - `CodeLlama-70B-Python-GPTQ`
- Model types and compatibility considerations:
  - Some models are modified with quantization, pruning, or other techniques for hardware compatibility
  - Models marked with `GGUF`, `GPTQ`, `GGML`, `AWQ`, and similar tags may require specific configurations or tweaking for proper functionality

## Exercise

- Create an account on [Hugging Face](https://huggingface.co/)
- Navigate to the [Model Hub](https://huggingface.co/models)
- Find a text generation model to download and experiment with
- Try to run the model locally using Hugging Face Transformers or Text Generation WebUI (or any similar tool), at your preference
