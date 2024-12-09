# Lesson 03: Multi-Modal AI Applications

After learning how to build and integrate AI applications, we'll delve into the realm of multimodal AI models. We'll learn how to integrate multimodal AI models using OpenAI APIs into our applications, enabling us to perform various generation tasks with a single model.

In this lesson, we'll explore the extraction of information from images using Computer Vision models. We'll examine how these models function and their application to various computer vision tasks.

We'll review how the versatile Transformers architecture, previously discussed for text and image generation, can be applied to image processing tasks.

This lesson delves deeper into the relationship between textual concepts (labels or captions) and visual concepts in images. We'll introduce the concept of **World Scope**, which offers a fascinating perspective on how information can be correlated in unprecedented ways when combined in sufficient volumes.

We'll examine the interplay between visual and textual concepts within multimodal models, laying the groundwork for our future study of image generation models capable of processing text-to-image tasks. There we'll study how the transformer architecture, previously used in text generation models, can also be applied to image generation. We'll explore the noising and denoising processes and how they've been adapted in recent years to create sophisticated image generation algorithms.

Additionally, we'll set up the Stable Diffusion WebUI application to easily load, manage, and run Image Generation AI models locally.

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

## Building Multimodal AI Applications

- Multimodal AI models overview
  - A single model to process text, images, and audio
  - Latency considerations
  - Function calling capabilities
  - Output format options
- The [Gemini model](https://cloud.google.com/use-cases/multimodal-ai)
- The [Claude 3 model family](https://www.anthropic.com/news/claude-3-family)
- The [GPT-4 vision-enabled models](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/)
- The [GPT-4o omni model](https://openai.com/index/hello-gpt-4o/)
- Building a [simple multimodal application](https://sdk.vercel.ai/docs/guides/multi-modal-chatbot)

## Building a Multi-Modal Chat Application

1. Create a sample project using the Vercel AI SDK

   ```bash
   npx create-next-app@latest multi-modal-chatbot
   ```

   - Pick all the default options

2. Navigate to the project directory

   ```bash
    cd multi-modal-chatbot
   ```

3. Install the dependencies

   ```bash
   npm install ai @ai-sdk/openai
   ```

4. Configure OpenAI API key to a local environment variable

   - Create a `.env.local` file in the root of the project

   - Add the OpenAI API key variable in the file by inserting this: `OPENAI_API_KEY=xxxxxxxxx`

   - Replace `xxxxxxxxx` with your OpenAI API key

5. Create a Route Handler at `app/api/chat/route.ts`:

   ```tsx
   import { openai } from "@ai-sdk/openai";
   import { convertToCoreMessages, streamText } from "ai";

   export async function POST(req: Request) {
     const { messages } = await req.json();

     const result = await streamText({
       model: openai("gpt-4o"),
       messages: convertToCoreMessages(messages),
     });

     return result.toDataStreamResponse();
   }
   ```

   - Use a model with multimodal vision capabilities, like [gpt-4o](https://platform.openai.com/docs/models/gpt-4o)

6. Open the `app/page.tsx` file

7. Add the chat component to the page:

   ```tsx
   "use client";

   import { useChat } from "ai/react";

   export default function Chat() {
     const { messages, input, handleInputChange, handleSubmit } = useChat();
     return (
       <div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
         {messages.map((m) => (
           <div key={m.id} className="whitespace-pre-wrap">
             {m.role === "user" ? "User: " : "AI: "}
             {m.content}
           </div>
         ))}

         <form
           onSubmit={handleSubmit}
           className="fixed bottom-0 w-full max-w-md mb-8 border border-gray-300 rounded shadow-xl"
         >
           <input
             className="w-full p-2"
             value={input}
             placeholder="Say something..."
             onChange={handleInputChange}
           />
         </form>
       </div>
     );
   }
   ```

8. Import and implement a `useState` and a `useRef` hook:

   ```tsx
   import { useState, useRef } from "react";

   export default function Chat() {
      const { messages, input, handleInputChange, handleSubmit } = useChat();

      const [files, setFiles] = useState<FileList | undefined>(undefined);
      const fileInputRef = useRef<HTMLInputElement>(null);

      ...
   ```

9. Add a `div` inside the message content to upload files:

   ```tsx
   <div>
     {m?.experimental_attachments
       ?.filter((attachment) => attachment?.contentType?.startsWith("image/"))
       .map((attachment, index) => (
         <img
           key={`${m.id}-${index}`}
           src={attachment.url}
           width={500}
           alt={attachment.name}
         />
       ))}
   </div>
   ```

10. Modify the `form` element to handle the file upload:

    ```tsx
    <form
      className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl space-y-2"
      onSubmit={(event) => {
        handleSubmit(event, {
          experimental_attachments: files,
        });

        setFiles(undefined);

        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }}
    >
      <input
        type="file"
        className=""
        onChange={(event) => {
          if (event.target.files) {
            setFiles(event.target.files);
          }
        }}
        multiple
        ref={fileInputRef}
      />
      <input
        className="w-full p-2 text-black"
        value={input}
        placeholder="Say something..."
        onChange={handleInputChange}
      />
    </form>
    ```  

11. Your code should look like this:

    ```tsx
    "use client";

    import { useChat } from "ai/react";
    import { useState, useRef } from "react";
    
    export default function Chat() {
      const { messages, input, handleInputChange, handleSubmit } = useChat();
    
      const [files, setFiles] = useState<FileList | undefined>(undefined);
      const fileInputRef = useRef<HTMLInputElement>(null);
      return (
        <div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
          {messages.map((m) => (
            <>
              <div key={m.id} className="whitespace-pre-wrap">
                {m.role === "user" ? "User: " : "AI: "}
                {m.content}
              </div>
              <div>
                {m?.experimental_attachments
                  ?.filter((attachment) =>
                    attachment?.contentType?.startsWith("image/")
                  )
                  .map((attachment, index) => (
                    <img
                      key={`${m.id}-${index}`}
                      src={attachment.url}
                      width={500}
                      alt={attachment.name}
                    />
                  ))}
              </div>
            </>
          ))}
    
          <form
            className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl space-y-2"
            onSubmit={(event) => {
              handleSubmit(event, {
                experimental_attachments: files,
              });
    
              setFiles(undefined);
    
              if (fileInputRef.current) {
                fileInputRef.current.value = "";
              }
            }}
          >
            <input
              type="file"
              className=""
              onChange={(event) => {
                if (event.target.files) {
                  setFiles(event.target.files);
                }
              }}
              multiple
              ref={fileInputRef}
            />
            <input
              className="w-full p-2 text-black"
              value={input}
              placeholder="Say something..."
              onChange={handleInputChange}
            />
          </form>
        </div>
      );
    }

    ```

12. Run the project

    ```bash
    npm run dev
    ```

13. Open the browser and navigate to <http://localhost:3000>

14. Use an image as a prompt to generate a text response

    - Upload an image to the chat
    - Ask a prompt related to the image

15. Hit **Enter** to send the message

16. Experiment with the chat application

## Computer Vision Models

- Computer Vision (CV)
  - CV is a field of artificial intelligence that enables computers to interpret and understand visual information from the world
  - It involves developing algorithms and models that can process, analyze, and extract meaningful information from digital images and videos
  - CV has applications in various domains, including autonomous vehicles, facial recognition, medical imaging, and robotics

- Model Training
  - Model training in CV involves feeding large datasets of labeled images or videos to machine learning algorithms
  - The algorithms learn to recognize patterns, features, and relationships within the visual data
  - During training, the model adjusts its internal parameters to minimize the difference between its predictions and the actual labels
  - This process typically involves techniques such as backpropagation and gradient descent to optimize the model's performance

- Inference
  - Inference in CV refers to the process of using a trained model to make predictions or decisions on new, unseen data
  - During inference, the model applies the knowledge it gained during training to analyze and interpret new images or videos
  - This stage is where the practical applications of CV models come into play, such as identifying objects in real-time video streams or classifying medical images

- Image Processing Tasks
  - Image processing tasks in CV often involve manipulating or analyzing images to extract useful information or enhance their quality
  - These tasks can include operations like filtering, edge detection, color correction, and image segmentation
  - In the context of modern AI models, these tasks can be initiated through prompts or instructions given to the model, allowing for more flexible and dynamic image processing capabilities

- Using Transformers for CV
  1. **Image Classification**
     - A process where an algorithm is trained to recognize and categorize images into predefined classes or labels
     - Example: Classifying an image as a "cat", "dog", or "car"

  2. **Image Segmentation**
     - Involves dividing an image into multiple segments or regions based on specific criteria, such as objects, boundaries, or pixel similarities

  3. **Video Classification**
     - Similar to image classification, but analyzes video frames to categorize the entire video or its segments into predefined classes
     - Considers temporal information and may involve recognizing actions, events, or behaviors over time
     - Example: Classifying a video as "sports," "news," or "entertainment"

  4. **Object Detection**
     - The task of identifying and localizing specific objects within an image or video frame
     - Involves drawing bounding boxes around the detected objects and assigning them corresponding labels
     - Example: Detecting and localizing cars, pedestrians, and traffic signs in an autonomous driving scenario

  5. **Zero-shot Detection**
     - Refers to the ability of a model to recognize and locate objects it has never seen during training
     - Achieved by using prior knowledge, such as the semantic relationship between objects or attributes, to infer the presence of unseen classes without explicit examples

  6. **Zero-shot Classification**
     - Similar to zero-shot detection but focuses on assigning labels to images or videos without having seen training examples of those specific classes
     - Relies on utilizing semantic information or descriptions of the unseen classes to make predictions
     - Example: Classifying an image as a "giraffe" based on its description, even though the model has never been trained on giraffe images

  7. **Single-shot and Few-shot Detection and Classification**
     - Refers to the ability of a model to detect or classify objects with minimal training examples
     - Single-shot aims to perform these tasks with a single training example per class, while few-shot uses a few examples per class
     - Useful for scenarios where collecting large amounts of labeled data is challenging or expensive

## Computer Vision with SAM 2

- The [SAM 2](https://ai.meta.com/blog/segment-anything-2/) is a state-of-the-art Computer Vision model for semantic segmentation of images and videos
- Key Features:
  - **Video Segmentation**: Enables object segmentation in videos, tracking across frames and handling occlusions
  - **Memory Mechanism**: Incorporates a memory encoder, bank, and attention module to store and utilize object information, enhancing user interaction throughout videos
  - **Streaming Architecture**: Processes video frames sequentially, allowing real-time segmentation of lengthy videos
  - **Multiple Mask Prediction**: Generates multiple possible masks for ambiguous images or video scenes
  - **Occlusion Prediction**: Improves handling of temporarily hidden or out-of-frame objects
  - **Enhanced Image Segmentation**: Outperforms the original SAM in image segmentation while excelling in video tasks
- Improvements:
  - Unified architecture for both image and video segmentation
  - Rapid video object segmentation
  - Versatile model capable of segmenting novel objects, adapting to unfamiliar visual contexts without retraining, and performing zero-shot segmentation on images containing objects outside its training set
  - Fine-tuning of segmentation results by inputting prompts for specific pixel areas
  - Superior performance across various image and video segmentation benchmarks
- Example implementation: [Segmenting Images with SAM 2](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb) Notebook

## Multimodal Models with Computer Vision

- Language alone is insufficient to provide comprehensive information about our universe to Machine Learning models
  - The [Experience Grounds Language](https://arxiv.org/abs/2004.10151) paper proposes an intriguing perspective on how these ML models "understand" the scope of our world

  > Language understanding research is held back by a failure to relate language to the physical world it describes and to the social interactions it facilitates. Despite the incredible effectiveness of language processing models to tackle tasks after being trained on text alone, successful linguistic communication relies on a shared experience of the world. It is this shared experience that makes utterances meaningful.

- The AI's "understanding" of concepts present in the training data can be categorized based on the types of "perceptions" fed to the model during training to correlate these concepts:
  - WS1: Corpora and Representations (Syntax and Semantics)
  - WS2: The Written World (Internet Data)
  - WS3: The Perceivable World (Sight and Sound)
  - WS4: Embodiment and Action (Physical Space and Interactions)
  - WS5: Social World (Cooperation)
- The [Contrastive Language-Image Pretraining](https://github.com/openai/CLIP) (CLIP) model from OpenAI is one of the [first models to combine text and vision](https://openai.com/index/clip/) to comprehend concepts in both text and image and even connect concepts between the two modalities
  - As it utilizes visual information in the training process itself, CLIP can be considered a [world scope three model](https://www.pinecone.io/learn/series/image-search/clip/)
- The relationship between Natural Language Processing (NLP) and Computer Vision (CV) concepts is made possible through the **Contrastive Pretraining** process applied to the CLIP model during training
- Example implementation: [CLIP Explainability](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb) Notebook

## Image Generation Models

- Generative AI for images
  - Generating pixels that form a coherent image for human perception
  - If an image is generated from a textual description including concepts `A`, `B` and `C`, a human observer should readily recognize the same `A`, `B` and `C` concepts in the generated image
- Evolution of Image Generation algorithms
  - Early approaches: Simple techniques like [cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton) (1940s) and [fractals](https://en.wikipedia.org/wiki/Fractal) (1975) for pattern generation
  - Procedural generation: Utilized in early [video games](https://en.wikipedia.org/wiki/Procedural_generation#Video_games) (1980s) for landscape and texture creation
  - [Markov Random Fields](https://en.wikipedia.org/wiki/Markov_random_field): Applied to texture synthesis in the 1990s
  - [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis): Employed for face generation and manipulation in the early 2000s
  - Non-parametric sampling: Techniques like [image quilting](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) (2001) for texture synthesis and image analogies
  - [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine): Used for learning image features and generation in the mid-2000s
  - [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661): Introduced in 2014, GANs utilize a generator and discriminator network to create realistic images
  - [Variational Autoencoders (VAEs)](https://arxiv.org/abs/1312.6114): Developed in 2013, VAEs learn to encode and decode images, enabling generation of new samples
  - [Pixel RNNs](https://arxiv.org/abs/1601.06759) and [Pixel CNNs](https://arxiv.org/abs/1606.05328): These models, introduced in 2015-2016, generate images pixel by pixel using recurrent or convolutional neural networks
  - [Deep Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/dbm.pdf): Proposed in 2009, these energy-based models can generate images by learning probability distributions over pixel values
  - Autoregressive models: [PixelCNN++](https://arxiv.org/abs/1701.05517) (2017) improved upon earlier pixel-based models for sequential image generation
- Image Generation with Transformers

  - The [transformer architecture](https://arxiv.org/abs/1706.03762) used in GPTs can also be implemented for image generation models
  - Transformers can be trained to relate visual, textual, or audio concepts, allowing for versatility in many AI models
  - [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929) adapt the transformer architecture specifically for image tasks, including generation
  - **Transformer-based models** like [DALL-E](https://openai.com/dall-e-2) and [Stable Diffusion](https://stability.ai/stable-diffusion) have achieved state-of-the-art results in image generation

### Overview of Image Generation Techniques

- Multi-modal Approach

  - Multi-modal AI applications process and generate data in various formats, including text, image, video, and audio
  - These models create connections between different types of data, enabling more versatile and comprehensive AI systems
  - Benefits of multi-modal approaches:
    - Enhanced contextual associations and relationships between different data types
    - Improved ability to generate more coherent and contextually relevant outputs
    - Potential for more natural and intuitive human-AI interactions
      - A non-technical user can relatively easily prompt for a good generation even without using **prompt engineering** techniques
  - Example: ChatGPT with DALL·E
    - Utilizes unsupervised learning to generate images from text descriptions
    - Employs transformer language models to learn the relationship between text and images from large datasets
    - Demonstrates the ability to create novel, creative images based on complex textual prompts
    - Showcases the potential for AI to bridge the gap between linguistic and visual representations
    - DALL·E is fully integrated with [ChatGPT](https://chat.openai.com/), allowing it to automatically generate tailored, detailed prompts for DALL·E 3 when prompted with an idea
      - Users can request adjustments to generated images with simple instructions

- Neural Network-based Models

  - Neural networks, traditionally used for text processing, have been adapted for image generation tasks
  - These models learn to represent and generate complex visual patterns through contrastive training on large datasets
  - Key architectures used in image generation:
    - **Generative Adversarial Networks (GANs)**:
      - Utilize a generator and discriminator network to create realistic images
      - The generator creates images, while the discriminator attempts to distinguish real from generated images
      - This adversarial process leads to increasingly realistic image generation
    - **Variational Autoencoders (VAEs)**:
      - Learn to encode images into a compressed latent space and then decode them back into images
      - Allow for generation of new samples by manipulating the latent space
      - Useful for tasks like image reconstruction and style transfer
    - **Convolutional Neural Networks (CNNs)**:
      - Particularly effective for image-related tasks due to their ability to capture spatial hierarchies
      - Can be combined with other architectures to improve image generation quality
    - **Recurrent Neural Networks (RNNs)**:
      - Used for sequential image generation tasks
      - Useful for generating images with temporal dependencies, such as video frames

- Transformer-based Models
  - Transformer architecture, originally designed for natural language processing, has been successfully applied to image generation
  - These models treat image generation as a sequence prediction task, similar to language modeling
  - Key features of transformer-based image generation models:
    - Self-attention mechanisms:
      - Allow the model to weigh the importance of different parts of the input data
      - Enable the model to capture global dependencies in the concepts present in the images
    - Ability to capture complex dependencies:
      - Enables generation of high-quality, coherent images
      - Allows for improved representation of long-range relationships in visual data
    - Scalability:
      - Can handle large datasets and long-range dependencies effectively
      - Allows for training on diverse and extensive image collections
    - Parallelization:
      - Allows for efficient computation and training
      - Enables faster generation of high-resolution images
  - Examples of transformer-based image generation models:
    - **DALL·E (OpenAI)**:
      - Generates images from textual descriptions
      - Demonstrates impressive creativity and representation of complex concepts
    - **Stable Diffusion (Stability AI)**:
      - Open-source model capable of generating high-quality images from text prompts
      - Known for its efficiency and ability to run on consumer hardware
    - **Imagen (Google)**:
      - Produces photorealistic images with strong text alignment
      - Showcases advanced capabilities in representing and rendering complex scenes
  - These models have achieved state-of-the-art results in image generation tasks, demonstrating the versatility and power of the transformer architecture
  - Ongoing research focuses on improving the efficiency, controllability, and ethical use of these powerful image generation models

### Image Generation Examples

- [DALL·E-2 and DALL·E-3](https://labs.openai.com/) integrated with [ChatGPT](https://chat.openai.com/)
- [DALL·E-3](https://openai.com/index/dall-e-3/) integrated with [Designer](https://www.bing.com/images/create) (formerly Bing Image Creator)
- [Midjourney](https://www.midjourney.com/showcase)
- [DreamStudio](https://beta.dreamstudio.ai/generate)
- [Stable Diffusion](https://stablediffusionweb.com/)
- [Canva Text to Image](https://www.canva.com/your-apps/text-to-image)
- [Adobe Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html)
- [Imagen by Google](https://imagen.research.google/)
- [Craiyon](https://www.craiyon.com/) (formerly DALL-E mini)
- [Picsart](https://picsart.com/create/editor)
- [Image-FX](https://aitestkitchen.withgoogle.com/tools/image-fx)

## Overview of Stable Diffusion for Image Generation

- Similarities with GPT Language Models (LLMs)
  - Underlying architecture and operational principles
  - Use of transformer-based generative architecture
- Key differences from GPT models
  - Adapted to handle visual data instead of textual data
  - Processes pixel or feature sequences rather than textual sequences
- Image generation process
  - Initiated with a text prompt or partially completed image
  - Input encoded into latent space representation
  - Transformer iteratively refines the encoded representation
    - Guided by patterns learned during training
  - Refinement continues until a coherent, contextually relevant image is generated
- Core concepts
  - **Latent space representation**: Compressed encoding of input data
  - **Iterative refinement**: Gradual improvement of the image through multiple steps
  - **Contextual relevance**: Alignment with the initial prompt or partial image
  - **Diffusion process**: Gradual denoising of a random noise input to generate an image
  - **Attention mechanisms**: Allowing the model to focus on relevant parts of the input
  - **Conditioning**: Incorporating text prompts or other inputs to guide image generation
- Capabilities
  - Generates images from textual descriptions or incomplete images
  - Demonstrates "understanding" of visual data and context
  - Infers pixels and features related to concepts in the prompt
    - Similar to how GPT infers tokens related to other tokens
- Generation process details
  - Starts from a random noise image
  - Predicts and adds details in each iteration
  - Decodes latent representation into a coherent, detailed image
- Advantages of the approach
  - Produces highly detailed and contextually relevant images
  - Leverages transformer's ability to handle complex dependencies and relationships
  - Applicable to both text-to-image and image-to-image tasks

## Generating Images with Stable Diffusion

- Overview of Stable Diffusion
  - **Stable diffusion**: A type of **generative model** using a **diffusion process** to generate images
    - **Generative model**: Machine learning models that create new data similar to training data
    - **Diffusion**: Process of gradually adding noise to data and learning to reverse it
  - Core concept: Iteratively apply transformations to a **random noise vector** to produce a coherent image
    - **Noise vector**: Randomly initialized values serving as the starting point
    - **Transformations**: Mathematical operations shaping the noise into an image
  - Each step moves the noise vector closer to resembling a realistic image
- Running a Diffusion Model
  - Process: Iteratively apply transformations to a random noise vector
  - Goal: Produce a coherent image
  - Method: Select transformations that guide the noise vector towards resembling a realistic image
  - Termination: Predetermined number of steps or when output meets quality criteria
- How it Works
  - Stable diffusion employs two main steps: **forward diffusion** and **reverse diffusion**
  1. Forward Diffusion (Noising Process)
     - Begins with an image, gradually adds **Gaussian noise** over steps until it becomes pure noise
     - Modeled as a **Markov chain**: Each step depends only on the previous one
     - Utilizes conditional Gaussian distributions
     - Transforms data distribution into a **tractable distribution** (e.g., isotropic Gaussian)
       - **Tractable distribution**: Easy to sample from and compute probabilities
       - **Isotropic Gaussian**: Multivariate normal distribution with independent variables and uniform variance
  2. Reverse Diffusion (Backward Denoising)
     - Reverses the noising process, converting noise back into an image
     - Employs a **neural network** to predict and subtract added noise at each step
     - Network is conditioned on factors such as current noise level and possibly text embeddings
       - **Embedding**: Dense vector representation of discrete data in continuous space
     - Iterative process: Begins from pure noise, gradually reduces noise to reconstruct an image
     - Output: Sample resembling a natural image or corresponding to text description
- Training the Model
  - Optimizes a **loss function**: Measures difference between predicted and actual added noise
  - **Loss function**: Guides optimization of model parameters
- Image Generation
  - Initiates from random Gaussian noise image
  - Iteratively denoises using learned reverse diffusion process

> The best way to understand visual AI generation is with visual examples. The [The Illustrated Stable Diffusion](http://jalammar.github.io/illustrated-stable-diffusion/) article is an excellent resource to understand this process visually.

## Using OpenAI Image Generation APIs

- OpenAI API endpoints for [Image Generation](https://platform.openai.com/docs/api-reference/images)
- Generating images with [DALL·E](https://platform.openai.com/docs/guides/images) from text prompts
  - DALL·E-2 vs DALL·E-3
    - DALL·E-2: Earlier version with "good" image quality but less coherence with complex prompts
    - DALL·E-3: Latest version with "improved" understanding of prompts, "enhanced" image quality, and more accurate text rendering
  - Quality
    - Options typically include 'standard' and 'hd' (high definition)
    - Higher quality settings produce more detailed images but may require longer generation times
    - This parameter is only supported for DALL·E-3
  - Size
    - Must be one of `256x256`, `512x512`, or `1024x1024` for DALL·E-2
    - Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for DALL·E-3
    - Pricing varies based on size
  - Styles
    - Can be either `vivid` or `natural`
    - Only available for DALL·E-3
- Generating images with other images as input
  - Edit
    - Prompt: Textual description of desired changes to the original image
    - Mask: Specifies which areas of the image to modify, allowing for targeted edits
  - Variations
    - Creates multiple versions of an input image while maintaining its core elements
    - Useful for exploring different artistic interpretations or styles

## Running Generative AI Models

- Generative AI models
  - Closed models
    - Proprietary models developed by companies like OpenAI, Anthropic, or Google
    - Often have restricted access and usage policies
    - Examples include DALL·E, Midjourney, and Imagen
  - Open Source (or Open Access) models
    - Publicly available models with more flexible usage terms
    - Can be downloaded, modified, and run locally
    - Examples include Stable Diffusion and open-source versions of CLIP
- Running models
  - Local execution: Running models on your own hardware
  - Cloud-based services: Using platforms like Google Colab or AWS to run models
  - API integration: Accessing models through web APIs provided by companies
- Using Hugging Face's [Stable Diffusion Pipelines](https://huggingface.co/docs/diffusers/quicktour)
  - Hardware requirements
    - GPU with at least 10GB VRAM for optimal performance
    - Can run on CPU, but significantly slower
  - Tutorial: [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)
    - Step-by-step guide to using Stable Diffusion in a cloud environment
  - Dependencies
    - Python 3.7+
    - PyTorch 1.7.0+
    - Transformers library
    - Diffusers library
  - [Installation](https://huggingface.co/docs/diffusers/installation)
    - Detailed official instructions for installing the Diffusers library and its dependencies
  - Usage
    - [Loading models and configuring schedulers](https://huggingface.co/docs/diffusers/using-diffusers/loading)
      - How to load pre-trained models and set up different sampling methods
    - [Running a pipeline](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)
      - Steps to execute the image generation process
  - Text-to-image Pipeline [parameters](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
    - Detailed explanation of various parameters that control the image generation process
    - Includes options for prompt, negative prompt, number of inference steps, and guidance scale

## Setting Up Stable Diffusion WebUI

- The [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) tool
  - Inspired the `Text Generation WebUI` we used previously
  - Built on [Gradio](https://www.gradio.app/) as well
  - Similar interface and usage
- Addressing the [Dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies)
  - Instructions for [NVIDIA](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)
  - Instructions for [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)
  - Instructions for [Intel](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon)
  - Instructions for [Apple](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
  - Options for running in [Docker Containers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Containers)
  - Options for running in [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)
- Instructions for [Installing and Running](https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#installation-and-running)
- Installing and running packages with [Stability Matrix](https://github.com/LykosAI/StabilityMatrix)
  - **Note**: You can install Stable Diffusion WebUI using the [Stability Matrix](https://github.com/LykosAI/StabilityMatrix) tool
  - Choose the correct distribution for your OS from the [Downloads page](https://lykos.ai/downloads)
  - Install the latest stable release
  - Run the tool and choose `+ Add Package` in the `Packages` tab
  - Select `Stable Diffusion WebUI` from the list and click `Install`
- Alternative [Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services) if your environment is incompatible with the dependencies
- Usage
  - Access the WebUI at <http://127.0.0.1:7860/> (default location)
  - Run with the `--api` flag to access the API at <http://127.0.0.1:7860/docs>
- Key [Features](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features)
  - Stable Diffusion: Loading, managing, and running stable diffusion models
  - Text to Image tasks: Generates images based on textual prompts provided by the user
  - Image to Image tasks: Transforms existing images based on text prompts or other images
  - Inpainting: Allows editing specific parts of an image while keeping the rest intact
  - Outpainting: Extends images beyond their original boundaries, creating seamless expansions
  - Image variations: Generates multiple versions of an image with slight variations
  - Resizing/Upscaling: Increases the resolution and quality of images without losing detail
- Using Models and Checkpoints
- Managing configurations
- Navigating the interface

## Using Model Checkpoints

- Downloading models
  - Using Stability Matrix to manage models
    - Checkpoint manager
    - Model Browser
      - CivitAI
      - Hugging Face
  - Using Stable Diffusion WebUI
    - From [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image&library=diffusers&sort=downloads) (filter for "Diffusers" under "Library")
      - You can easily download models using a [model downloader extension](https://github.com/Iyashinouta/sd-model-downloader)
        - Download and install using the `Extensions` tab
          - Use `Install from URL` tab
          - Paste the extension URL in the `URL for extension's git repository` input field
          - Click the `Install` button
          - Go to the `Installed` tab
          - Click the `Apply and restart UI` button
        - After installing, go to the `Model Downloader` tab
    - From [CivitAI](https://civitai.com/models?tag=base+model) (filter for "Checkpoints" under "Model Types")
      - A secure way to download models from CivitAI is to use the [CivitAI Helper Extension](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper)
        - Download and install using the `Extensions` tab
          - Use `Install from URL` tab
          - Paste the extension URL in the `URL for extension's git repository` input field
          - Click the `Install` button
          - Go to the `Installed` tab
          - Click the `Apply and restart UI` button
        - Go to `Civitai Helper` tab
          - Use the `Download Model` pane to get model information from a link and download it
            - You can use the `Block NSFW Level Above` filter to conveniently filter out images tagged as NSFW
    - From other websites like [PromptHero](https://prompthero.com/ai-models/text-to-image) and [Reddit](https://www.reddit.com/r/StableDiffusion/wiki/models/)
- Models
  - High resolution model [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - **Note**: Not suitable for generating readable text, faces, or objects with compositional structure
  - Standard resolution photorealistic model [Photon](https://civitai.com/models/84728/photon)
    - **Note**: Has some issues with rendering hands and fingers
  - Versatile model for digital art, drawings, anime, and painting [GhostMix](https://civitai.com/models/36520?modelVersionId=76907)
    - Requires the use of LoRAs for optimal results
    - **Note**: Not very effective for photorealistic images or generating images of "common" persons or scenes
  - Versatile high resolution model [JuggernautXL](https://civitai.com/models/133005/juggernaut-xl)
    - Good generalist model that effectively utilizes light/shadow effects
- Choosing models
  - Model file extensions
    - `ckpt` files are models where Python can load serialized data, but they are potentially dangerous as they can execute arbitrary code
    - `safetensors` files are not affected by the vulnerability of `ckpt` files and are preferred when available
  - Model weights
    - `EMAonly` models use only the Exponential Moving Average of the weights
      - These models are smaller and faster to process, requiring less VRAM to run
      - Ideal for generating images, but not suitable for fine-tuning
    - `Full` models use all the weights, including the EMA and the non-EMA
      - These models are larger and slower to process, requiring more VRAM to run
      - Ideal for training new models, but not suitable for generating images
    - `Pruned` models have had unnecessary/irrelevant weights removed
      - These models have fewer weights to process, making them faster to run
      - **Note**: Removing weights can sometimes negatively impact the model's accuracy for certain prompts, especially for terms not well-represented in the training data
  - LoRAs (Low-Rank Adaptation)
    - LoRAs are techniques used to adapt a model to specific characteristics or datasets
    - Some models include this within the checkpoint, enhancing the quality of generated images but potentially decreasing compatibility with other LoRAs that could be used in conjunction with the model
  - VAEs (Variational autoencoders)
    - Some models require a VAE to be used in conjunction with them to properly generate images
      - The VAE encodes images into a latent space that the model uses during training
      - At generation time, the model decodes points from the latent space back into images
      - Without the matching VAE, the model can't properly reconstruct the images
    - If a checkpoint specifies a certain VAE requirement, you must use that VAE to achieve proper image generation; otherwise, the results will be suboptimal
  - Model categories
    - `Base` models are the most common and versatile, and are the best choice for use as a base for fine-tuning
    - `XL` models are high-resolution models that can generate more detailed images but are slower to process and require more VRAM
    - `Anime` models specialize in generating anime-style images
    - `Cartoon` models specialize in generating cartoon-style images
    - `Art` models specialize in generating digital art, drawings, paintings, and other artistic styles
    - `Photorealistic` models specialize in generating images that resemble real photographs
    - `Portrait` models specialize in generating images of people
    - `Hybrid` models combine two or more of the previous categories and are the best choice for tasks requiring a mix of styles or variety in the generated images
    - [Model Classification](https://civitai.com/articles/1939/an-attempt-at-classification-of-models-sd-v15) article

## Generating Images with Stable Diffusion WebUI

- **Selecting a Model (checkpoint)**

  - The starting model is the "classic" `legacy/stable-diffusion-v1-5` [Model](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  - Specifically, the version used is `v1-5-pruned-emaonly`, ideal for inference (generating images) due to lower VRAM usage
    - The "full" `v1-5-pruned` version is suitable for fine-tuning but uses more VRAM
      - `EMAonly` refers to the use of only the Exponential Moving Average (EMA) of the weights, a technique used to stabilize model training
      - The "full" version includes all weights (EMA and non-EMA), using more VRAM but providing more flexibility for fine-tuning
    - "Pruned" indicates the removal of unnecessary/irrelevant weights, making the model smaller and faster to process with minimal performance/precision loss
  - In summary, use the starting model for image generation, but consider changing it before attempting fine-tuning

- **Passing a prompt**

  - Similar to the `Text Generation WebUI`, you can pass a prompt to the model to generate an image
  - We'll explore this in practice later

- **Understanding CLIP**

  - Every textual prompt must be encoded using Contrastive Language-Image Pre-Training before being used to generate images
  - CLIP is a neural network that learns to associate images and text, enabling it to "understand" image content and text meaning
  - Most base SD models have a 75 (or 77) token limit for input
    - Prompts exceeding this limit will be truncated
    - Stable Diffusion WebUI allows for "infinite" prompt length by [breaking the prompt into chunks](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#infinite-prompt-length)

- **Prompting techniques**
  - Generally, more specific and objective prompts yield "better" results
  - As with text-to-text models, prompts are only effective when their contents relate to the model's training data
  - Many models allow for configuring specific [attention/emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis) of each term in the prompt
  - These text-to-image models may also follow instructions to avoid content specified in [Negative Prompts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt)

### Configurations

- **Hardware configurations**

  - Edit the `webui-user.sh` (Linux/MacOS) or `webui-user.bat` (Windows) file at the `COMMANDLINE_ARGS` line
    - Using [Low VRAM Mode](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#4gb-videocard-support)
    - Using [CPU only](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-And-Settings#running-on-cpu)
- Configuring the image generation options
  - **Width and height**: The dimensions of the output image
    - Example: Set width to 512 and height to 384 for a 4:3 aspect ratio, or width to 512 and height to 768 for a 2:3 portrait aspect ratio
    - **Important**: Due to the nature of Stable Diffusion models, even a single pixel change in resolution can significantly alter the generated image
      - For different resolutions:
        1. Generate images at the model's native resolution (typically 512 pixels)
        2. Use the high-resolution fix extension to enhance quality for larger sizes
      - Alternative: Manipulate the [Image Generation Seed](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#sampling-method-selection) to maintain similar outputs across resolutions
  - **Batch size**: Number of images generated simultaneously
    - Consumes more VRAM but offers practical advantages:
      - Generates multiple variations at once
      - Allows for efficient selection of "satisfactory" results
      - Reduces the need for repeated single-image generation
- Configuring the sampling options
  - **Sampling steps**: The fixed number of iterations to generate the image
    - Generally, more steps lead to more detailed images but increase generation time
    - **Note**: After a certain point, additional steps may not noticeably improve image quality and could potentially degrade it in some algorithms
  - **Sampling method**: The algorithm used to sample the image at each step
    - Types of sampling methods:
      - Old-School ODE solvers:
        - Numerical methods for solving Ordinary Differential Equations (ODEs)
          - ODEs: Mathematical equations describing the rate of change of a quantity with respect to another
        - Examples:
          - `Euler`: Simplest solver, approximates ODE solution using finite differences
            - Finite differences: Approximate derivatives using discrete intervals
            - While simple, it can be inaccurate for complex systems
          - `Heun`: More accurate but slower version of Euler, uses second-order Runge-Kutta method
            - Runge-Kutta methods: A family of iterative methods for approximating solutions of ODEs
            - Second-order: Uses two evaluations per step, improving accuracy over Euler
          - `LMS` (Linear multi-step method): Uses multiple previous points to calculate the next solution point
            - Improves accuracy by considering the solution's history
            - Can be more stable for certain types of ODEs
      - Ancestral samplers:
        - Stochastic samplers that add noise at each step, introducing randomness
          - Stochastic: Involving random probability distribution or pattern
            - In this context, it means the sampling process isn't deterministic
            - Each run can produce slightly different results, even with the same inputs
        - Examples: `Euler a`, `DPM2 a`, `DPM++ 2S a`, `DPM++ 2S a Karras`
        - **Characteristics**:
          - Faster processing
            - The added randomness can help the sampler explore the solution space more efficiently
          - May produce noisier results
            - The stochastic nature can introduce artifacts or inconsistencies in the output
          - Outputs can vary significantly between runs
            - This variability can be both a strength (for generating diverse outputs) and a weakness (for reproducibility)
      - `Karras` noise schedule:
        - Developed by Tero Karras (creator of StyleGAN)
        - **Key features**:
          - Carefully controls noise input to improve image variety and realism
            - Noise is added in a structured way to maintain image coherence while introducing variation
          - Initializes and learns noise scaling factors during training
            - This allows the model to adapt the noise levels to different parts of the image generation process
          - Generally slower but produces more consistent, less noisy results
            - The careful noise control requires more computation but often yields higher quality outputs
      - `DDIM` (Denoising Diffusion Implicit Models):
        - Variant of denoising diffusion probabilistic models
        - Uses a non-Markovian reverse process
          - Non-Markovian: Future states depend on more than just the current state
            - This allows the model to consider longer-term dependencies in the generation process
            - Can lead to more coherent and globally consistent images
        - **Advantages**:
          - Generates high-quality samples in fewer steps
            - The non-Markovian nature allows for more efficient sampling
          - Relatively simple and easy to train
            - Despite its sophistication, the model architecture is straightforward to implement and optimize
      - `PLMS` (Pseudo Linear Multi-Step):
        - Variant of the linear multi-step method
        - Uses a probabilistic approach, potentially faster than original LMS
      - Diffusion Probabilistic Models:
        - Generate new data samples through iterative denoising
        - Examples:
          - `DPM`: Basic version using a Markov chain
            - Markov chain: A sequence of possible events where the probability of each depends only on the state in the previous event
              - In DPM, each denoising step only depends on the immediately previous state
              - Simple but can be limited in capturing long-range dependencies
          - `DPM-2`: Improved version using a non-Markovian process
            - Allows for consideration of multiple previous states
            - Can capture more complex relationships in the data
          - `DPM++`: Further improvement with new parameterization
      - `UniPC` (Unified Predictor-Corrector Framework):
        - Training-free framework for fast sampling of diffusion models
        - **Key features**:
          - Model-agnostic design: Can work with various types of models
            - Not tied to a specific architecture, making it versatile across different diffusion models
          - Supports various DPM types and sampling conditions
            - Can adapt to different diffusion processes and initial conditions
          - Faster convergence due to increased accuracy order
  - Comparing sampling methods
    - Consider both performance and quality when selecting a method
    - Refer to comprehensive comparisons:
      - [Performance comparison](https://stable-diffusion-art.com/samplers/)
      - [Quality comparison](https://learn.rundiffusion.com/sampling-methods/)
- Classifier Free Guidance (`CFG`) scale
  - This parameter controls how closely the model adheres to your text prompt during the sampling steps
  - Lower values (close to 1) allow for more creative freedom, while higher values (above 10) are more restrictive and may affect image quality
  - **Caution**: Extremely low CFG values may cause Stable Diffusion to "ignore" your prompt, while excessively high values can lead to oversaturated colors
    - [CFG Scale Comparison Article](https://www.artstation.com/blogs/kaddoura/pBPo/stable-diffusion-samplers)
- Seed
  - The initial value used to generate the random tensor in the latent space
  - This value enables the reproduction of identical or highly similar images from the same prompt and settings
  - Variational seeds can yield intriguing results, simulating exploration of the latent space between two defined seeds
    - This technique can create a "blending" effect between two images, with intermediate generations appearing as a gradual transformation
- Built-in extensions
  - **Restore faces**: Scripts and tools to replace distorted faces with more realistic ones
  - **Tiling**: A technique to modify image borders for seamless repetition in a grid
  - **High resolution fix**: Methods and tools for image upscaling
    - **Note**: Generating high-resolution images is challenging for most models, often resulting in lower quality compared to native resolutions
    - **Best practice**: Generate images at the model's native resolution (typically 512 pixels), then use the high resolution fix to enhance quality
- Additional functionalities
  - The Stable Diffusion WebUI offers various image generation tasks beyond text-to-image conversion:
    - **Image to Image**: Generate new images based on existing ones
    - **Sketch**: Create images from rough drawings or annotated images
    - **Inpainting**: Generate images to replace specific areas within larger images
- **CLIP (Contrastive Language-Image Pretraining)**: Tool for extracting potential prompts describing an image

## Prompting Techniques for Image Generation

- **Overall considerations**
  - Optimal techniques for each model type
  - Positive prompts
  - Negative prompts
  - Presence or absence of embeddings in the training data
- Positive prompt guidelines
  - Structure: Clearly specify the major defining elements in the image
    - Subject: "Generate an image of {object}" where {object} is the primary element you want to create
      - Be as specific as possible, considering the representation of the term in the training data
      - Sometimes using a more general term is preferable to specific ones
      - If desired, add an action: "Generate an image of {object} {action}", e.g., "standing", "running", "flying"
    - Specify form, quantity, and adjectives: "Generate an image of {quantity} {quality} {adjective} {object}"
      - Note: Not all models are trained to handle these types of prompts effectively
        - Prompts like "two big sweet red apples" might yield unexpected results in many models
  - Context: Specify image composition elements, such as settings, style, and artistic form
    - Indicate if it's a portrait, landscape, still life, etc.
    - Specify if it should resemble a photograph, painting, drawing, sculpture, etc.
    - Define composition details like background, lighting, colors, etc.
      - Add specifications like "close-up", "far away", "profile", etc.
    - Depending on the model's training, specify the image style, e.g., "impressionist", "surreal", "realistic"
      - If a model is trained on a specific artist's work, you can mention the artist's name
      - Caution: If the model's training data lacks references to a style, results may differ significantly from expectations
  - Refinements: Some models can handle more specific instructions to add nuance to the image
    - Specify the mood, e.g., "happy", "sad", "scary"
    - Indicate time of day, weather, season, etc.
    - Control lighting, color scheme, temperature, detail level, realism level, etc.
  - Tip: To test if a model "understands" a term, try generating an image with just that word as the prompt
    - Example: Before generating "A happy futuristic landscape picture of Darth Vader riding a blue unicorn", try generating "Darth Vader" and "Unicorn" separately to assess their representation in the outputs
- Negative prompts
  - Some models accept negative prompts to instruct certain elements or aspects to be avoided in the generated image
  - Applying negative prompts may reduce the likelihood of certain flaws or deformations but can also limit the overall quality of the generated images
  - Different subjects may require different sets of negative prompts for optimal results
    - For living creatures, consider including: deformed, ugly, too many fingers, too many limbs
    - For objects, use more general terms like: duplicated, blurry, pixelated, low quality, out of frame, cropped
    - For faces, depending on the pose and close-up, consider: poorly rendered face, deformed, disfigured, long neck, ugly
    - For landscapes, include: blurry, pixelated, out of frame, cropped, text, watermark, low quality, poorly drawn
- Adjusting [Attention/Emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)
  - Select specific words in your prompt to emphasize or de-emphasize by surrounding them with special characters
    - The model's tokenizer interprets these special characters as attention markers, adjusting word probabilities accordingly
    - Each model may use different special characters, so consult the documentation for the specific model you're using
    - SD models used by the WebUI employ two types of special characters: `()` and `[]`
      - `(word)` increases attention to the word by a factor of 1.1
      - `((word))` increases attention by a factor of 1.21 (1.1 \* 1.1)
      - `[word]` decreases attention by a factor of 1.1
      - `(word:1.5)` increases attention by a factor of 1.5
      - `(word:0.25)` decreases attention by a factor of 4 (1 / 0.25)
      - To use these characters literally in a prompt, escape them with a backslash (`\`), e.g., `\(word\)` or `\[word\]`
    - Other models may use different special characters like `{}`, `++`, `--`, etc. Consult the documentation for your specific model/tooling
- Prompt examples
  - [Prompt Templates](https://github.com/Dalabad/stable-diffusion-prompt-templates) repository
  - [Prompt Presets](https://openart.ai/presets) from OpenArt
- The [StyleSelectorXL](https://github.com/ahgsql/StyleSelectorXL) extension

## Group Exercise

- Form a group of 2-5 people around you
- Create a Github repository for your group
- Start with a blank page with a text input and a button
- On click of the button, generate an image based on the prompt in the text input
  - Experiment with OpenAI API and local models
  - Display the image in the page
- Commit the changes to the repository
- Deploy the application to Vercel
