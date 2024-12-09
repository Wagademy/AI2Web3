# Lesson 04: Decentralized AI

In this lesson we'll cover the subject of decentralized AI, how to build AI Agents that can interact with the world and how to monetize them using tokens.

 We are going to introduce the concept of Web3 and smart contracts, and how these technologies can be used to enhance the capabilities of AI models, bringing benefits from peer to peer networking, token economic incentives, decentralized infrastructures and much more.

## Prerequisites

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

## Introduction to Web3

Web3 represents the next evolution of the internet, characterized by decentralization, blockchain technology, and token-based economics. It provides a framework for creating trustless, permissionless applications that can interact with AI Agents.

### Key Concepts of Web3

- Decentralization and distributed systems
- Blockchain technology and its role in Web3
- Cryptocurrency and tokenomics
- Decentralized applications (dApps)
- Interoperability and cross-chain communication

### Web3 Infrastructure

- Ethereum and other smart contract platforms
- Layer 2 scaling solutions
- Decentralized storage (e.g., IPFS, Filecoin)
- Decentralized identity systems

### Web3 Development Tools

- Web3.js and Ethers.js libraries
- Truffle and Hardhat development frameworks
- MetaMask and other wallet integrations
- IPFS and Pinata for decentralized file storage

## Smart Contracts

Smart contracts are self-executing contracts with the terms of the agreement directly written into code. They run on blockchain networks and can interact with AI Agents to create complex, automated systems.

### Understanding Smart Contracts

- Definition and characteristics of smart contracts
- How smart contracts work on blockchain networks
- Benefits and limitations of smart contracts
- Popular smart contract languages (e.g., Solidity, Vyper)

## Decentralized AI

The capabilities of Generative AI models can be extended to decentralized applications by integrating them with smart contracts. This allows for creating many powerful combinations, such as financially autonomous agents, AI-powered marketplaces, dataset sharing for distributed training, peer-to-peer GPU computing markets, and more.

## ORA Protocol

- [ORA Protocol](https://ora.io/)

### Optimistic ML

- The [opML paper](https://arxiv.org/abs/2401.17555)
- Open-source framework for verifying ML inference onchain
- Similar to optimistic rollups
- Example [opML powered AI](https://www.ora.io/app/opml/openlm)

### Implementing Decentralized AI Model Inferences

- The [OAO repository](https://github.com/ora-io/OAO)
- Implementing the `IAIOracle.sol` interface
- Building smart contracts [with ORA's AI Oracle](https://docs.ora.io/doc/ai-oracle/ai-oracle/build-with-ai-oracle)
- Handling the [Callback gas limit estimation](https://docs.ora.io/doc/ai-oracle/ai-oracle/callback-gas-limit-estimation) for each model ID
- [Reference list](https://docs.ora.io/doc/ai-oracle/ai-oracle/references) for models and addresses for different networks

### Sample Implementation

- Open the `Prompt.sol` file in the `contracts` folder in [Remix IDE](https://remix.ethereum.org/)
- Add `callbackGasLimit[11] = 5_000_000;` in the `constructor()` function
- Deploy the contract to the `sepolia` network using your `injected provider` as environment
- Pass the AI Oracle Proxy address to the `Prompt` contract deployment function
  - The address for the `sepolia` network is `0x0A0f4321214BB6C7811dD8a71cF587bdaF03f0A0`
- Call the `estimateFee` function passing the model ID `11` to check the fee for the inference
- Copy the value and set it as the transaction value (in wei)
- Call the `calculateAIResult` targeting the model ID `11` with your prompt
- When the transaction is confirmed, it might take a little while until the callback is executed and the result is available
  - When the callback is executed, the result is stored in the storage of the `Prompt` contract
  - The `promp` function can be called to retrieve the result, by passing the model ID and prompt text again

### Extending the AI Oracle

A troll is sitting in front of the contract guarding the vault.

- The deployer of the smart contract will deposit ETH into the contract
  - The value is locked while a boolean is set to `false`, the default state
- The deployer will set the riddle for the troll
- Anyone can attempt to solve the riddle by calling the `solveRiddle` function
  - If the riddle is solved correctly, the boolean is set to `true` and the deposit can be withdrawn
  - For solving the riddle, the string passed should be considered a valid solution by the AI model
- Example riddle: `Thirty white horses on a red hill, First they champ, Then they stamp, Then they stand still.`
- Using a [starting template](https://hardhat.org/hardhat-runner/docs/getting-started) for the `Lock.ts` contract
- Implementing the AI Oracle interface

  ```solidity
  import "./interfaces/IAIOracle.sol";
  import "./AIOracleCallbackReceiver.sol";

  contract Lock is AIOracleCallbackReceiver {
      // modelId => callback gasLimit
      mapping(uint256 => uint64) public callbackGasLimit;

      struct AIOracleRequest {
          address sender;
          uint256 modelId;
          bytes input;
          bytes output;
      }

      // requestId => AIOracleRequest
      mapping(uint256 => AIOracleRequest) public requests;
  ```

- Implementing the riddle logic

  ```solidity
  bool public unlockFunds;
  string public riddle;
  address public winner;
  string public constant PROMPT_CONFIG =
      "I am going to give you a riddle marked as RIDDLE and a proposed solution marked as SOLUTION. If the solution provided is acceptable for the riddle, answer with the word CORRECT, and nothing else. If the solution provided is not acceptable, answer WRONG, and nothing else.";
  uint256 public constant MODEL_ID = 11;
  ```

- Implementing the `constructor()` function

  ```solidity
  constructor(IAIOracle _aiOracle, string memory _riddle)
        payable
        AIOracleCallbackReceiver(_aiOracle)
    {
        callbackGasLimit[MODEL_ID] = 5_000_000;
        riddle = _riddle;
    }
  ```

- Implementing the `solveRiddle()` function

  ```solidity
  function solveRiddle(string calldata solution) external payable {
     require(winner == address(0), "Winner is already set!");
     bytes memory fullPrompt = abi.encodePacked(
         PROMPT_CONFIG,
         " RIDDLE: ",
         riddle,
         " SOLUTION: ",
         solution
     );
     uint256 requestId = aiOracle.requestCallback{value: msg.value}(
         MODEL_ID,
         fullPrompt,
         address(this),
         callbackGasLimit[MODEL_ID],
         ""
     );
     AIOracleRequest storage request = requests[requestId];
     request.input = fullPrompt;
     request.sender = msg.sender;
     request.modelId = MODEL_ID;
  }
  ```

- Implementing the oracle callback function

  ```solidity
  // the callback function, only the AI Oracle can call this function
    function aiOracleCallback(
        uint256 requestId,
        bytes calldata output,
        bytes calldata callbackData
    ) external override onlyAIOracleCallback {
        // since we do not set the callbackData in this example, the callbackData should be empty
        require(winner == address(0), "Winner is already set!");
        AIOracleRequest storage request = requests[requestId];
        require(request.sender != address(0), "request not exists");
        request.output = output;
        if (keccak256(output) == keccak256(bytes("CORRECT"))) {
            winner = request.sender;
            unlockFunds = true;
        }
  }
  ```

- Modifying the `withdraw()` function to allow the winner to withdraw the funds when the riddle is solved

  ```solidity
  function withdraw() public {
      require(unlockFunds, "You can't withdraw yet");
      emit Withdrawal(address(this).balance, block.timestamp);
      payable(winner).transfer(address(this).balance);
  }   
  ```

- Deploy the contract to the `sepolia` network using your `injected provider` as environment
- Call the `solveRiddle()` function passing the correct solution
- When the transaction is confirmed and the callback is executed, call the `withdraw()` function to withdraw the funds

### Initial Model Offerings

- Model Ownership ([ERC-7641 Intrinsic RevShare Token](https://ethereum-magicians.org/t/erc-7641-intrinsic-revshare-token/18999)) + Inference Asset (eg. [ERC-7007 Verifiable AI-Generated Content Token](https://github.com/AIGC-NFT/ERCs/blob/master/ERCS/erc-7007.md))
- IMO launches an ERC-20 token (more specifically, ERC-7641 Intrinsic RevShare Token) of any AI model to capture its long-term value
- Anyone who purchases the token becomes one of the owners of this AI model
- Token holders share revenue of the IMO AI model
- The [IMO launch blog post](https://mirror.xyz/orablog.eth/xYMD27tN23ppbKCluB9faytF_W6M1hKXTuKcfkm3D50) and the [first IMO implementation](https://mirror.xyz/orablog.eth/GSjMm-qC4WWsduGqCISSvA1IxicJbyRDES_bl7-Tt2o)

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

## Group Exercise

- Form a group of 2-5 people around you
- Create a Github repository for your group
- Start with the Agent code from the previous exercise
  - Deploy a smart contract for your token to be used with the agent
  - Implement the function for the agent to read your balance of tokens
  - Ask the agent about your balance of tokens in your wallet
    - Bonus: Make the agent answer differently according to the balance of tokens in your wallet
- Commit the changes to the repository
- Deploy the application to Vercel
