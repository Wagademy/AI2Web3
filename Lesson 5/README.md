# Lesson 05: Decentralized AI Agents

In the past lessons, we explored how Large Language Models (LLMs) can generate textual responses to user prompts. These models can follow instructions and call functions while fulfilling user requests. However, after generating an answer, the process typically ends, and the model remains dormant until the next prompt arrives.

Today, we'll delve into the concept of AI Agents, which can continuously run and invoke functions until a specific condition or stop trigger is met. Beyond merely generating textual answers, AI Agents can be assigned tasks and/or goals to accomplish. They attempt to discover the optimal approach to achieve desired results by utilizing LLMs and other AI resources to guide the process.

We are going to learn how AI Agents can operate like autonomous entities that can perform tasks without explicit instructions. We'll implement agents that can interact with the environment, make decisions, and take actions based on the information they receive.

We'll explore how these AI Agents differ from simple programmed scripts and how the inference process used to generate textual answers can also guide the "thought" process of these agents.

By the end of this this lesson we'll cover all you need to know about building and deploying your project, from idea to production. We'll explain the process and techniques that we use in the professionals projects that we build in our Software Factory, and how these have helped many previous students to build their projects with success, winning grants, hackathon bounties, prizes, and even getting jobs at top tech companies.

## Prerequisites

- A group of 2-5 people from previous days
- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Installation of Python tools on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency in using `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Proficiency in using `venv` to create and manage virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Node.js installed on your device
  - [Node.js](https://nodejs.org/en/download/)
- Proficiency with `npm` and `npx` commands
  - Documentation: [npm](https://docs.npmjs.com/)
  - Documentation: [npx](https://www.npmjs.com/package/npx)
- Understanding of `npm install` and managing the `node_modules` folder
  - Documentation: [npm install](https://docs.npmjs.com/cli/v10/commands/npm-install)
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for cloning repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Basic knowledge of JavaScript programming language syntax
  - [JavaScript official tutorial](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/javascript/)
- Basic knowledge of TypeScript programming language syntax
  - [TypeScript official tutorial](https://www.typescriptlang.org/docs/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/typescript/)
- An account at [OpenAI Platform](https://platform.openai.com/)
  - To run API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account

## AI Agents

- The AI Agent "reasoning" process
  - **Differs from static programs** built with coded scripts and well-defined methods
  - AI Agents attempt to "figure out" the best approach to achieve desired results using LLMs and other AI resources
  - A conventional program relies on scripted logic to respond to user input, while an AI Agent uses LLMs to determine the optimal course of action
  - LLM limitations also apply to AI Agents' "reasoning" process:
    - Struggle with complex tasks requiring extensive information processing
    - Difficulty with tasks demanding specificity and accuracy due to potential hallucinations or mistakes
    - Challenges with tasks requiring creativity, as they may struggle to generate original ideas beyond training data
  - AI Agents can be assigned tasks and/or goals, and they strive to discover the best method to achieve desired outcomes using LLMs and other AI resources
- Agent Functions
  - **Core components** of an AI Agent
  - Can be considered the "brain" of an AI Agent
  - Triggered by users or other functions when fulfilling a prompt
- Agent programs
  - Provide infrastructure for hosting and managing agent function invocations and fulfillments
  - Offer an interface to interact with agent functions by:
    - Initiating processes
    - Defining and managing goals
    - Monitoring the agent's progress and status
    - Orchestrating the agent's execution
    - Handling agent's failures and errors
    - Providing feedback and information to the user
  - **Context/state management** is crucial for the agent's execution:
    - Allows tracking of the agent's progress and status
    - Provides necessary information to agent functions
    - Manages all data that needs to be handled/carried between different function invocations
    - Essential because LLMs cannot independently maintain the agent's state between function calls
  - **Performance management** is a significant role of the agent program:
    - Inference operations can be computationally intensive, especially with larger models
    - Ensures each step is executed efficiently to deliver expected results
    - Optimizes task execution to balance performance and accuracy

## Types of Agents

- **Reactive Agents**
  - Most basic type, capable of responding only to direct user input
  - No extended "reasoning" or planning beyond answering user input
- **Reflective Agents**
  - Simple reflex agents
    - Operate based on predefined rules and immediate data
    - Limited to reacting within given event-condition-action rules
    - Precise and efficient but unable to handle undefined situations
  - Model-based reflex agents
    - More sophisticated decision-making mechanism
    - Utilize Large Language Models (LLMs) to infer probable outcomes based on current state and goals
    - Attempt to evaluate consequences of action paths before determining task execution order
    - Leverage available data to construct an "emulated model of the world" for decision support
- **Rule-based Agents**
  - AI agents with more robust "reasoning" capabilities
  - Analyze environmental data and compare approaches to achieve desired outcomes
  - Goal-based agents aim to select the most efficient path
  - Agent program employs Natural Language Processing (NLP) for "reasoning" based on available information and goals
- **Utility-based Agents**
  - Similar to goal-based agents but focused on optimizing specific outcomes
  - Tasks defined to maximize or minimize measurable units of perceivable value within obtainable information
  - Examples: optimizing text style or length, identifying best item prices within a specified time frame
- **Hierarchical Agents**
  - Comprise multiple AI entities organized in a layered structure
  - Top-tier agents decompose complex objectives into manageable subtasks, which are then assigned to lower-level agents
  - Each individual agent functions autonomously and reports progress to its overseeing agent
    - Lower-tier agents can be simpler implementations, such as reactive or reflective agents
  - Higher-level agents consolidate outcomes and coordinate the efforts of their subordinates to ensure overall goal achievement

## Building Agents on top of OpenAI Features

- OpenAI Assistants features
  - [Code Interpreter](https://platform.openai.com/docs/assistants/tools/code-interpreter)
  - [File Search](https://platform.openai.com/docs/assistants/tools/file-search)
    - [Vector Stores](https://platform.openai.com/docs/assistants/tools/file-search/vector-stores)
  - [Function Calling](https://platform.openai.com/docs/assistants/tools/function-calling)
    - [Function Call Workflow](https://platform.openai.com/docs/guides/function-calling)
    - Using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) for [Function Calling](https://platform.openai.com/docs/guides/function-calling/function-calling-with-structured-outputs)
- Building Agents with OpenAI Tools using [LangChain](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/)
- Building Agents with OpenAI features using [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/)

## Automation Process

- Goals
  - Agent Programs are designed to receive specific instructions to scope one or more measurable goals that govern the agent's execution
    - These goals determine the **task planning** process, where the agent composes a set of actions it "believes" necessary to achieve the desired results, according to how the NLP models can "reason" about concepts related to the passed goals and the agent's training to trigger each programmed function
    - The model's training allows linking actionable tasks to specific concepts or sets of concepts contained in the output of an inference operation from a prompt
  - Each goal is defined as a desirable outcome that the agent can "understand" sufficiently to evaluate at any given moment whether that goal has been achieved
- Information acquisition
  - The agent must gather information from various sources to achieve the desired results, and the agent program must provide the necessary tools for this purpose
  - In some cases, all required information is already accessible to the Agent Program, simplifying the process to selecting available information from the source and passing it to subsequent tasks
  - In other instances, when information is not readily available, the agent may need to trigger intermediary tasks for function invocations before proceeding with the main task initially chosen to progress towards the desired goal
  - These tasks could include:
    - Using the `web_search` function to search for information on the web
    - Employing the `file_search` function to search for information in local files
    - Utilizing the `vector_search` function to search for information in the vector database
- Implementing tasks
  - When the agent program determines that the goal has not yet been achieved but sufficient data exists to execute the next task, it methodically implements the next task by invoking the agent functions programmed to handle the task with the provided data
  - Upon task completion, the program removes it from the list of planned tasks and proceeds to the next task until the goal is achieved
  - If the agent exhausts its task list, the program can be configured to either devise an alternative task plan or halt, report progress to the user, and terminate execution
- "Reasoning"
  - The "reasoning" process in AI agents typically follows a decision flow with nodes and edges:
    - Nodes represent agents and tools:
      - Agent nodes: Different types of agents (e.g., reactive, reflective, rule-based) that process information and make decisions
      - Tool nodes: Functions or APIs that agents can use to gather information or perform actions
    - Edges represent conditions and checks:
      - Condition edges: Logical conditions that determine which path the decision process should take
      - Check edges: Validation steps to ensure the output of a node meets certain criteria before proceeding
  - The decision process flow involves traversing nodes and edges while updating the execution state of the agent program at each step
    - The agent program must provide a mechanism to update the agent program's state at each step, enabling it to track the agent's progress and status, and supply necessary information to agent functions
  - This flow of actions between nodes and edges forms a **graph**
    - The graph can be evaluated before execution to estimate possible paths
    - This evaluation helps optimize the agent's goal by:
      - Filtering out paths that don't lead to goal achievement
      - Eliminating paths that are significantly longer than shorter alternatives
  - The process of **routing** choices between nodes and edges forms the **decision process** of the agent
    - Since LLMs cannot truly "reason" about how to draw these routes based on available information in the state and goal criteria, the "emulated reasoning" process involves passing available node choices to the LLM and then running a completion task to generate probable outputs

## Setting up an AI Agent

- Installing necessary tools
- Configuring the environment
- Creating the agent program
- Building agent functions
- Integrating the agent with the agent program
- Testing the agent

## Using Functions to Enhance "Reasoning"

- How to [teach reasoning](https://openai.com/index/learning-to-reason-with-llms/) on top of non-reasoning text-generation processes
  - OpenAI [GPT-4o1](https://openai.com/index/introducing-openai-o1-preview/) example
- Automating the Chain of Thought prompt generation with agents
  - Amazon Web Services [Auto-CoT](https://arxiv.org/pdf/2210.03493) example
- Variations for Chain of Thought
  - [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm) (ToT)
  - [Graph of Thoughts](https://github.com/spcl/graph-of-thoughts) (GoT)
  - [Algorithm of Thoughts](https://github.com/kyegomez/Algorithm-Of-Thoughts) (AoT)

## Agentic RAG

- Retrieval Agents
  - File search functions
  - Web search functions
  - Chunking data
  - Embeddings
  - Vector search
- Integrating Query Engines into Agent tasks
- Multi-tool invocation
- Implementing Query Transformations
  - Decision-making for information retrieval
- Multi-step queries

## Monetizing AI Agents with Tokens

AI Agents can be designed to handle monetary transactions using digital tokens, which are often implemented on blockchain networks. This capability opens up new possibilities for creating autonomous economic systems and incentivizing AI-driven services.

### Introduction to Tokenization

- Definition of tokens in the context of blockchain and cryptocurrency
- Types of tokens:
  - Utility tokens
  - Security tokens
  - Governance tokens
  - Non-fungible tokens (NFTs)
- Benefits of using tokens for AI Agent transactions:
  - Programmable money
  - Transparency and auditability
  - Global accessibility
  - Fractional ownership

## Decentralized AI Agents tools

- Building AI Agents with token enabled functions
- Managing wallets and transactions
- Building smart contracts
- Deploying decentralized AI Agents

## Creating a Token-Enabled AI Agent

Let's create a simple AI Agent that can interact with a smart contract to perform token transfers. We'll use the [Coinbase AgentKit](https://www.coinbase.com/en-br/developer-platform/discover/launches/introducing-agentkit) for creating the agent and implementing the blockchain interaction.

1. Create an account and get an API key at [Coinbase Developer Platform](https://portal.cdp.coinbase.com/projects/api-keys)
2. Create an API ket in the [OpenAI Platform](https://platform.openai.com/api-keys) and fund your account with at least $5 or so for testing
3. Fork the template from [AgentKit Replit Template](https://replit.com/@CoinbaseDev/CDP-AgentKit#README.md)
   - Once forked, you'll have your own version of the project to modify
4. Configure Your Environment

   - Click on `Tools` in the left sidebar
   - Click on `Secrets` in under the `Workspace Features` section
   - Add the following secrets:

   ```text
   CDP_API_KEY_NAME=your_cdp_key_name
   CDP_API_KEY_PRIVATE_KEY=your_cdp_private_key
   OPENAI_API_KEY=your_openai_key
   NETWORK_ID="base-sepolia" # Optional, defaults to base-sepolia
   ```

5. Run the Agent
   - Click on `Run` in the top panel

## Deploying Applications to a Decentralized Infrastructure

- Hosting applications
- Frontend and backend integration
- Databases
- Web3 integration

## Next Steps

- Join the community
- Attend future events
- Enroll in more bootcamps
- Attend hackathons
- Apply for grants/prizes
- Apply for jobs
- Enjoy the sponsor's perks for this bootcamp
