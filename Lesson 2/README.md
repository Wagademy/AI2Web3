# Lesson 02: Building AI Applications

In this lesson, we will explore the process of creating AI applications using web technologies and AI models. We'll begin by examining the concept of a web application and learning how to construct a basic web page.

Our journey starts with the fundamentals of setting up a web development environment and crafting a simple web page using HTML, CSS, and JavaScript. We'll experiment with common web development tasks and then investigate how frameworks can help us build more complex applications efficiently.

Subsequently, we'll delve into using the Next.js React framework to construct a more sophisticated web application. We'll discover how Next.js can assist in building efficient and scalable applications while abstracting away many of the complexities associated with web technologies.

In this lesson we will leverage the use of code-generation tools that use the capabilities of LLMs to write working pieces of code.

We'll also learn to leverage the Vercel NextJS AI SDK to accelerate development, taking advantage of its numerous "out-of-the-box" features.

Finally, we'll clone and run a sample project from Vercel, using it as a foundation to construct our own AI application.

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

## What is a Web Application?

- Definition and characteristics of Web Applications
- HTML, CSS, and JavaScript: The building blocks
- Opening a web page in a browser

## Creating an Example Index.html File

1. Create a new file named `index.html` in a separated folder in your computer

2. Insert some content in the file

   ```html
   <!DOCTYPE html>
   <html>
     <head>
       <title>Page Title</title>
     </head>
     <body></body>
   </html>
   ```

   - Every HTML page has a basic structure with texts and _hypertext markups_ (HTML tags).

   - The `<!DOCTYPE html>` tag is used to tell the browser that the document is an HTML document

     - If this is not included, the browser sometimes might get lost on how to interpret the document

   - The `<html>` tag is the root element of an HTML page, where all other elements are nested inside it

   - The `<head>` tag contains meta-information about the document, such as its title tag and links to stylesheets and scripts

     - The `<title>` tag is used to specify the title of the document, which is displayed in the browser's title bar or tab

   - The `<body>` tag contains the visible page content

     - This is where you put the text, images, and other elements that you want to display on the page

3. You can replace the text "Page Title" with the title of your page (e.g., `"My First Web Page"`)

4. Now, let's add some content to your page

   - Inside the `<body>` tags, you can add many different elements to your page, such as headings, paragraphs, images, and links

   - For example, let's add a heading and a paragraph by writing the following:

   ```html
   <h1>Welcome to My Web Page</h1>
   <p>This is my first paragraph.</p>
   ```

5. Save your changes and open the page with your web browser

6. The final code should look like this:

   ```html
   <!DOCTYPE html>
   <html>
     <head>
       <title>Hello world</title>
     </head>
     <body>
       <h1>Welcome to My Web Page</h1>
       <p>This is my first paragraph.</p>
     </body>
   </html>
   ```

7. Feel free to add more elements to your page

   - Here are some examples:

     - Images: You can add images with the `<img>` tag. The source of the image is specified in the `src` attribute

       - For example, `<img src="image.jpg">` would display the image named `"image.jpg"`

     - Links: You can create clickable links with the `<a>` tag. The destination of the link is specified in the `href` attribute

       - For example, `<a href="https://www.example.com">Visit Example.com</a>` would create a link to `"example.com"`

     - Lists: You can create lists with the `<ul>` (unordered/bullet list), `<ol>` (ordered/numbered list), and `<li>` (list item) tags

       - For example, `<ul><li>First item</li><li>Second item</li></ul>` would create a bullet list with two items

   - Experiment by adding these elements to your page

     - After adding each one, save your file and reload the page on your browser to see what it does

     - This is a great way to learn in practice how different HTML tags work

> You can learn more about the syntax and features of HTML at the [MDN Web Documentation](https://developer.mozilla.org/en-US/docs/Web/HTML) or with free tutorials like [W3Schools](https://www.w3schools.com/html/)

## Setting Up a Web Development Environment

- Essential tools for Web Development
  - **Text Editor**: A tool for writing and editing code, such as [Visual Studio Code](https://code.visualstudio.com/), [Sublime Text](https://www.sublimetext.com/), or [Atom](https://atom-editor.cc/)
  - **Web Browser**: A software application for accessing information on the World Wide Web, such as [Google Chrome](https://www.google.com/chrome/), [Mozilla Firefox](https://www.mozilla.org/en-US/firefox/new/), or [Microsoft Edge](https://www.microsoft.com/en-us/edge)
  - **Terminal**: A command-line interface for interacting with the operating system, such as [Windows Command Prompt](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/windows-commands), [Windows PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview), [macOS Terminal](https://support.apple.com/guide/terminal/welcome/mac), or [Linux Terminal](https://ubuntu.com/tutorials/command-line-for-beginners)
  - **Version Control System**: A tool for tracking code changes and collaborating with other developers, such as [Git](https://git-scm.com/)
  - **JavaScript Runtime**: An environment that allows you to run JavaScript code outside of a web browser, such as [Node.js](https://nodejs.org/en/download/)
  - **Node Package Manager**: A package manager for JavaScript that allows you to install and manage dependencies for your projects, such as [npm](https://docs.npmjs.com/)
    - Alternatives include [Yarn](https://yarnpkg.com/), [pnpm](https://pnpm.io/), [bun](https://bun.sh/), or other package managers
- Using Node.js
  - Utilizing a package manager
  - The package.json file
    - Installing dependencies
    - Running scripts
  - The `node_modules` folder
  - Folder structure
- Using Git
  - Cloning a repository
  - Committing changes
  - Pushing changes
  - Pulling changes
  - Branching and merging
- Creating a Web Application project
- Utilizing starter-kits and frameworks

## Coding Using Next.js

- Framework Structure
  - Recent versions of Next.js allow project creation with two major structures: [Pages router](https://nextjs.org/docs/pages/building-your-application/routing/pages-and-layouts) or [App router](https://nextjs.org/docs/app/building-your-application/routing/pages-and-layouts)
    - The Pages router is the more common structure, where each file in the `pages` directory corresponds to a route in the application
    - The App router is a more recent and slightly more complex structure, where routes are governed by a JavaScript API instead of being implicitly defined by the file system
- Utilizing TypeScript and TSX
  - [TypeScript](https://www.typescriptlang.org/docs/) is a superset of JavaScript that adds static typing, providing type checking and code completion features that help catch errors early in the development process
    - TypeScript is a powerful tool for building large-scale applications, enforcing strict type checking and providing a more robust development experience
  - [TSX](https://www.typescriptlang.org/docs/handbook/jsx.html) is a syntax extension for TypeScript allowing developers to write JSX (JavaScript XML) in TypeScript files
    - JSX is a syntax extension for JavaScript enabling developers to write HTML-like code directly in their JavaScript files
    - TSX can be used to create [React Components](https://react.dev/learn/typescript) that are processed and rendered dynamically when the application is compiled/bundled for deployment
- Creating Components
  - Components are the building blocks of a React application, representing reusable and independent parts of the user interface
    - Components can be functional or class-based, containing their own logic, state, and lifecycle methods
  - Components can be created using the `function` keyword or the `class` keyword
    - Functional components are simpler and more concise, while class-based components offer more features and flexibility
  - Components can receive data through props, passed from parent components
    - Props are read-only and cannot be modified by the component
  - Components can also have internal state, managed using the `useState` hook
    - State allows components to store and update data locally
- Using Hooks
  - [Hooks](https://react.dev/reference/react/hooks) are functions allowing functional components to use state and other React features
    - Hooks provide a way to reuse stateful logic across components, facilitating the sharing and management of stateful logic in a React application
  - Common hooks include:
    - `useState`: Allows components to manage local state
    - `useEffect`: Enables components to perform side effects, such as data fetching and DOM manipulation
    - `useContext`: Allows components to access context values
    - `useRef`: Enables components to create mutable references to DOM elements
  - There are [recommended rules](https://react.dev/reference/rules/rules-of-hooks) for using hooks in React applications to avoid bugs and ensure proper functionality
- Implementing Tailwind CSS
  - [Tailwind CSS](https://tailwindcss.com/docs) is a utility-first CSS framework streamlining the styling process by providing pre-defined utility classes for direct use in HTML markup
    - Tailwind CSS is designed to be highly customizable and extendable, allowing developers to create unique and responsive designs without writing custom CSS
  - Tailwind CSS classes can be [applied directly to HTML elements](https://tailwindcss.com/docs/utility-first) for styling
    - Classes are used to apply styles such as colors, fonts, spacing, and layout to elements
    - These classes can be [reused for many elements](https://tailwindcss.com/docs/reusing-styles) and combined to create complex layouts and designs
  - Tailwind CSS provides utility classes for [responsive design](https://tailwindcss.com/docs/responsive-design), enabling developers to create layouts that adapt to different screen sizes
    - Responsive classes can be used to apply different styles based on screen size, such as hiding elements on mobile devices or changing the layout on larger screens
  - You can [customize your styles](https://tailwindcss.com/docs/adding-custom-styles) in Tailwind CSS by editing the configuration file and adding custom utility classes
    - Customizing Tailwind CSS allows you to create a unique design system for your entire application and tailor the framework to your specific needs

## Building an Application Using Next.js

1. Create a new folder for your projects:

   ```bash
   mkdir my-projects
   cd my-projects
   ```

   > Pick a _safe_ location on your computer to store your projects. You can use the `Documents`, `Desktop`, or any other folder you prefer.

2. Create a new NextJS project using the following command:

   ```bash
   npx create-next-app my-next-app
   ```

   - You can give any name to your project by replacing `my-next-app` with your preferred name
   - Pick all the default options when prompted
     - ✔ Would you like to use TypeScript? … No / **Yes**
     - ✔ Would you like to use ESLint? … No / **Yes**
     - ✔ Would you like to use Tailwind CSS? … No / **Yes**
     - ✔ Would you like to use `src/` directory? … **No** / Yes
     - ✔ Would you like to use App Router? (recommended) … No / **Yes**
     - ✔ Would you like to customize the default import alias (@/\*)? … **No** / Yes

3. Navigate to the newly created project folder:

   ```bash
   cd my-next-app
   ```

4. Start the development server:

   ```bash
   npm run dev
   ```

5. Open your browser and navigate to `http://localhost:3000` to see your NextJS project running

6. Open the `app/page.tsx` file in your editor to make changes to the home page

7. Replace the existing code in `app/page.tsx` with the following:

   ```tsx
   export default function Home() {
     return (
       <>
         <h1>Hello World</h1>
         <p>This is a test</p>
       </>
     );
   }
   ```

8. Save the file and refresh your browser to see your changes

9. Apply some styling to the page using the `Tailwind CSS` classes

   ```tsx
   export default function Home() {
     return (
       <>
         <div className="flex flex-col items-center justify-center min-h-screen">
           <h1 className="text-4xl font-bold mb-4 text-blue-600">
             Hello World
           </h1>
           <p className="text-xl text-gray-500">This is a test</p>
         </div>
       </>
     );
   }
   ```

10. Create some page elements

    ```tsx
    export default function Home() {
      return (
        <>
          <div className="flex flex-col items-center justify-center min-h-screen">
            <div className="flex flex-col items-center justify-center">
              <h1 className="text-4xl font-bold mb-4 text-blue-600">
                Hello World
              </h1>
              <p className="text-xl text-gray-500">This is a test</p>
            </div>
            <div className="flex flex-row items-center justify-center py-4">
              <input
                type="text"
                placeholder="Enter your name"
                className="bg-gray-200 p-2 rounded-md mr-2"
              />
              <button className="bg-blue-600 text-white p-2 rounded-md">
                Submit
              </button>
            </div>
          </div>
        </>
      );
    }
    ```

11. Import some functionalities from the `react` library:

    ```tsx
    "use client";
    import { useState } from "react";
    ```

12. Create some dynamic content in the page:

    ```tsx
    export default function Home() {
      const [name, setName] = useState("");
      const [showGreeting, setShowGreeting] = useState(false);

      const handleSubmit = () => {
        setShowGreeting(true);
      };

      return (
        <>
          <div className="flex flex-col items-center justify-center min-h-screen">
            <div className="flex flex-col items-center justify-center">
              <h1 className="text-4xl font-bold mb-4 text-blue-600">
                Hello World
              </h1>
              <p className="text-xl text-gray-500">This is a test</p>
            </div>
            <div className="flex flex-row items-center justify-center py-4">
              <input
                type="text"
                placeholder="Enter your name"
                className="bg-gray-200 p-2 rounded-md mr-2 text-black"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
              <button
                className="bg-blue-600 text-white p-2 rounded-md"
                onClick={handleSubmit}
              >
                Submit
              </button>
            </div>
            {showGreeting && (
              <div className="mt-4 p-4 bg-gray-100 rounded-md w-full max-w-2xl">
                <p className="text-3xl font-bold text-black">Hello {name}</p>
                <textarea
                  className="w-full h-40 mt-4 p-2 text-lg text-black bg-white border border-gray-300 rounded-md resize-none"
                  placeholder="Enter your message here..."
                ></textarea>
              </div>
            )}
          </div>
        </>
      );
    }
    ```

## Implementing AI Features in Applications

### OpenAI APIs

- API Functionality
  - API calls enable remote execution of operations on servers over the internet, offering several advantages:
    - Leveraging substantial computational power not available on local devices
    - Processing large datasets that exceed local storage capabilities
    - Handling sensitive data (e.g., credentials, personal information) securely on remote servers
    - Protecting sensitive operations from client-side exposure
- Interacting with APIs over HTTP
  - APIs serve as essential interfaces for accessing and manipulating web service resources
  - In Python, the widely-used 'requests' library facilitates HTTP interactions:
    - Constructing HTTP requests:
      - Specify request method (GET, POST, PUT, DELETE, etc.)
      - Define API endpoint URL
      - Example: `response = requests.get('https://api.example.com/data')`
    - Processing API responses:
      - Retrieve data from response body
      - Access metadata (HTTP status codes, headers)
      - Example: `print(response.text)`
    - Implementing authentication:
      - Include credentials in request headers
      - Example:

        ```python
        headers = {'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}
        response = requests.get('https://api.example.com/data', headers=headers)
        ```

    - Handling errors:
      - Check response status codes
      - Implement appropriate error handling
      - Example:

        ```python
        if response.status_code == 200:
            print('Request was successful')
        else:
            print('Error:', response.status_code)
        ```

  - Benefits of API-based processing:
    - Execution of resource-intensive tasks on powerful remote servers
    - Secure handling of sensitive data without local exposure
    - Efficient transfer of processed results to the client

This approach enables developers to leverage remote computational resources and data securely, while maintaining a lightweight client-side application.

- OpenAI API
  - The OpenAI API provides access to a variety of powerful artificial intelligence models trained on large datasets, including text generation, image generation, and natural language recognition models. Developers can integrate AI capabilities into their applications, websites, and systems by accessing models through specific API endpoints.
  - Since the use of this computation is billed, OpenAI needs to identify and authorize (or deny) each person/agent making requests to the API
    - This is achieved by assigning a unique **secret key** or set of keys, which are specific to each user and must be included in every request made to the API
  - Authentication to the OpenAI API is performed through the use of an **API key**. Each user is assigned a unique API key that must be included in every request made to the API. The API key is typically passed as an **HTTP header** in the request or as a query parameter.
  - Each request made to the API incurs costs; that is, each computation performed on the OpenAI API is charged to the user
  - Costs may vary based on the type of model used, the size of the request and response, and the amount of compute power and resources utilized
    - The models have different costs depending on the amount of data (tokens, pixels, etc.) they process
    - The [Pricing](https://openai.com/api/pricing/) page has the most up-to-date information on the costs of the different models and tasks
- Endpoints
  - Each endpoint available in the OpenAI API offers specific functionality to meet developers' needs
  - There isn't a single endpoint that can be used to access all the models and capabilities of the OpenAI services
    - Instead, there are several endpoints, each providing access to a specific model or set of models, and each with its own set of capabilities and requirements
  - The most common endpoints are **chat** for text generation tasks, **audio** for speech recognition and creation, **images** for image generation and helpers for passing images to chat completion endpoints, and **fine_tuning** for creating and managing fine-tuning jobs and models
    - There are many other useful endpoints for tasks such as creating assistants, moderating text content, calculating text embeddings, and more
    - For a comprehensive list of features, refer to the "Capabilities" section of the [OpenAI Platform Documentation](https://platform.openai.com/docs/overview) and the "Endpoints" section of the [API Reference](https://platform.openai.com/docs/api-reference/introduction)
- Text Generation
  - The **chat** endpoint in the OpenAI API allows users to request text generation based on the provided input, which is useful for text autocompletion, content generation, and language translation
  - [Chat Completions](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) tasks require passing at least a **model definition** and a **message** in the **request body**
    - In the body, you can pass many other parameters such as model configurations for **temperature** and **max_tokens**, and request options for **stream** and **user**
  - The response object will contain the generated text inside an object called **choices**, as well as additional metadata such as the **model** used, the **system_fingerprint** of the GPT version, and the **usage** statistics of the API request
- Image Generation
  - The **image** endpoint in the OpenAI API allows users to request the generation of images based on a given input. Users can provide a description or visual input, and the API generates corresponding images using specific artificial intelligence models designed for this purpose.
- Vision
  - OpenAI's **chat** endpoint allows passing one or multiple images as content in the request body, under the **content** object, by giving it a type of **image_url**
    - The provided images can then be used when the model is generating a response by first being processed through an _image recognition_ process, and then being used as _context_ for the _text generation_ process
      - According to the [documentation](https://platform.openai.com/docs/guides/vision), the model excels at answering general questions about the contents of images, but it's not yet optimized to answer detailed questions about the location of specific objects in an image
        - For example, you can ask about the color of a car or suggest dinner ideas based on the contents of your fridge, but if you show it an image of a room and ask where the chair is, it may not answer the question accurately
  - You can control the **detail** parameter to specify how the model processes the image and generates its textual understanding
    - You can provide three options: **low**, **high**, or **auto**
    - By default, the model will use the **auto** setting, which will examine the image input size and decide whether to use the **low** or **high** setting
      - Using **low** will enable the "low res" mode, where the model receives a low-resolution 512px x 512px version of the image and represents the image with a budget of 65 tokens
        - This allows the API to return faster responses and consume fewer input tokens for use cases that don't require high detail
      - Using **high** will enable "high res" mode, which first allows the model to see the low-res image and then creates detailed crops of input images as 512px squares based on the input image size
        - Each of the detailed crops uses twice the token budget (65 tokens) for a total of 129 tokens

### Implementing OpenAI API Calls

- The [OpenAI Python API library](https://github.com/openai/openai-python)
- The [OpenAI Typescript API library](https://github.com/openai/openai-node)
- Environment setup
- Authentication
- Calling the endpoints
- Handling the responses

## Creating a Simple Chat Page

1. Open the project created in the previous exercise in a text editor

2. Edit your `page.tsx` file to include the following code:

   ```tsx
   "use client";

   import React from "react";

   const Home = () => {
     const message = "Hello World!";

     return (
       <main className="min-h-screen bg-gray-900 py-6 flex flex-col justify-center sm:py-12">
         <h1 className="text-4xl font-bold text-center text-gray-100 mb-8">
           Chat Page
         </h1>
         <section className="max-w-3xl mx-auto w-full">
           <div className="bg-gray-800 shadow-lg rounded px-8 pt-6 pb-8 mb-4">
             <p className="text-xl text-gray-300">{message}</p>
           </div>
         </section>
       </main>
     );
   };

   export default Home;
   ```

3. Save the file and navigate to `http://localhost:3000` in your browser to view the changes

4. Stop the server by pressing `Ctrl+C` or `Cmd+C` in your terminal

5. Install the `openai` [npm package](https://www.npmjs.com/package/openai) by running `npm install openai` in your terminal

6. Create a `.env` file in the root of your project and add your OpenAI API key

   ```text
   OPENAI_API_KEY=sk-...
   ```

7. Create a folder called `api` inside the `app` folder the root of your project

8. Create a folder named `chat` inside the `api` folder

9. Create a file named `route.ts` inside the `chat` folder

10. Paste the following code into the `route.ts` file:

    ```typescript
    import OpenAI from "openai";
    import { NextResponse } from "next/server";

    const openai = new OpenAI();

    export const runtime = "edge";

    export async function POST(req: Request) {
      const { messages } = await req.json();

      const response = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages,
      });

      return NextResponse.json({
        content: response.choices[0].message.content,
      });
    }
    ```

11. Modify the UI with the chat components:

    - In VSCode, open the file 'page.tsx' inside the folder 'app'

    - Replace the existing code with the following:

    ```typescript
    "use client";
    import { useState } from "react";

    const Home = () => {
      const [message, setMessage] = useState("");
      const [response, setResponse] = useState("");

      const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            messages: [{ role: "user", content: message }],
          }),
        });
        const data = await res.json();
        setResponse(data.content);
        setMessage("");
      };

      return (
        <main className="min-h-screen bg-gray-900 py-6 flex flex-col justify-center sm:py-12">
          <h1 className="text-4xl font-bold text-center text-gray-100 mb-8">
            Chat Page
          </h1>
          <section className="max-w-3xl mx-auto w-full">
            <div className="bg-gray-800 shadow-lg rounded px-8 pt-6 pb-8 mb-4">
              {!response && (
                <form
                  onSubmit={handleSubmit}
                  className="flex flex-col space-y-4"
                >
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Enter your message"
                    className="px-3 py-2 bg-gray-700 text-white rounded"
                  />
                  <button
                    type="submit"
                    className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
                  >
                    Send
                  </button>
                </form>
              )}
              {response && (
                <div className="mt-4 p-3 bg-gray-700 text-white rounded">
                  <p>{response}</p>
                </div>
              )}
            </div>
          </section>
        </main>
      );
    };

    export default Home;
    ```

12. Run the application: Now that everything is set up, you can start your application

    - On your terminal, run the following command:

    `npm run dev`

    - If everything is okay, you can now access the application at <http://localhost:3000/>

## Deploying the Application to a Hosting Service

- Web application hosting

  - Post-development, applications need deployment to make them accessible online
    - Static websites (HTML, CSS, JS) can utilize services like [GitHub Pages](https://pages.github.com/), [Netlify](https://www.netlify.com/), or file servers like [S3](https://aws.amazon.com/s3/) and [IPFS](https://ipfs.io/)
    - Server-Side Rendering (SSR) or Serverless Functions require services like [Vercel](https://vercel.com/) for server-side processing and API endpoints
  - Numerous free and paid hosting services are available
    - Selection depends on application requirements, budget, expected traffic, needed features, and desired developer control
    - For educational purposes, we'll use Vercel, which offers a suitable free tier for most projects

- CI/CD and DevOps for web applications

  - Automating deployment for code changes
    - Continuous Integration/Continuous Deployment (CI/CD) tools like [GitHub Actions](https://docs.github.com/en/actions) and [Vercel CLI](https://vercel.com/docs/cli) streamline this process
    - These tools create pipelines for automatic building and deployment upon code pushes
  - Vercel's free plan enables direct deployment from public GitHub repositories
    - Simplifies deployment by connecting the repository to Vercel for automatic updates
    - Vercel CLI allows command-line deployments, useful for non-GitHub hosted applications
  - Environment variables require separate configuration on Vercel

    - Set in the [Environment Variables](https://vercel.com/docs/environment-variables) section of project settings
    - Used for API keys, URLs, and sensitive information that shouldn't be hardcoded

## Deploying the Application to Vercel

1. If you don't have a Vercel account, you can create a free (_Hobby_) one [here](https://vercel.com/signup)
2. Install the [Vercel CLI](https://vercel.com/docs/cli) by running the following command in your terminal:

   ```bash
   npm install -g vercel
   ```

3. Link your project to Vercel by running the following command in your terminal:

   ```bash
   vercel link
   ```

   - Send **yes** to continue
   - Connect your Vercel account to the CLI
   - Choose the scope of the project (personal or organization)
   - Choose **N** for linking the repository to an existing project
   - Pick any project name you like
   - Input the path to the source of your project (should be `./`)
   - Vercel CLI should detect the framework as **Next.js** and set up the correct settings
   - Choose **N** for modifying the project's settings

4. Add the environment variables to the Vercel environments

   ```bash
   vercel env add OPENAI_API_KEY
   ```

   - Paste your OpenAI API key as the value for the `OPENAI_API_KEY` variable when prompted
   - Select all environments by pressing **Space** for each of the options
   - Hit **Enter** to confirm

5. Run the Vercel CLI deploy command in your terminal in the root of your project's folder to deploy your project:

   ```bash
   vercel
   ```

   - The project should take a little while to build and deploy

   - After the deployment is complete, you will receive a URL to access your project

   - The preview URL should require authentication unless you change the project's [Deployment Protection Settings](https://vercel.com/docs/security/deployment-protection) settings

6. Use the `--prod` flag to deploy the project to production for public usage

   - Projects deployed to _preview_ environments are not publicly accessible unless you change the project's [Deployment Protection Settings](https://vercel.com/docs/security/deployment-protection)

7. Follow the instructions in the terminal to deploy your project to Vercel

8. Pick the default options for NextJS deployment

9. Do not override the default settings

10. Once the deployment is complete, you will receive a URL to access your project

11. You may also use the Vercel dashboard to manage your project and your deployments

    - If you face errors due to the API keys, check the [Environment Variables](https://vercel.com/docs/deployments/environments) section of the project settings

## AI Code Generation

To expedite frontend code creation for our web application, we can leverage Generative AI Tools for code generation.

Tools like ChatGPT and Copilot can not only enhance developer efficiency but also generate substantial code segments that serve as starting points for software projects.

While the accuracy of these tools isn't perfect, and the generated code may require adjustments and improvements, it often provides a valuable foundation for developers to "understand" feature or functionality implementation.

Some models are specifically designed and/or optimized for code generation. Unlike the general-purpose LLMs we've been using, these models are trained on code repositories and can generate code snippets based on input prompts. Users still need to integrate these snippets to form complete code, but the generated content can serve as a solid starting point for projects or new features. Examples include [Codex](https://openai.com/index/openai-codex/) from OpenAI, [CodeT5](https://github.com/salesforce/CodeT5) from Salesforce, and [Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/) from Meta.

Other models/tools are designed for versatility, capable of generating code snippets, providing instructions, explaining code, suggesting fixes, proposing tests, adding documentation, and more. These models can detect code context and generate the most appropriate suggestions based on user requests. Examples include [Copilot](https://copilot.github.com/) (powered by the Codex model), [TabNine](https://www.tabnine.com/), and [Replit](https://replit.com/). These tools can generate code snippets based on your IDE context while coding.

Some tools are designed to automatically generate and assemble entire applications based on simple descriptions of desired layout and functionality. These can be particularly helpful for developers creating frontend code for web applications.

A standout tool for this purpose is [v0](https://v0.dev/) from Vercel. `v0` is a web tool that generates frontend code based on simple descriptions of desired layout and functionality.

Another valuable tool is [MakeReal](https://makereal.tldraw.com/) from TLDraw. `MakeReal` is a web tool that generates frontend code based on simple descriptions and rough sketches of desired layout and elements.

These tools can significantly assist developers in creating frontend code for web applications and can be utilized to generate code for this lesson's exercises.

> Be advised that the generated code may require significant adjustments and improvements to function properly. Never rely blindly on AI-generated code.

## Code Generation Models

- Training and fine-tuning techniques for code generation
- Instruction-following capabilities
- Retrieval and context evaluation methodologies
- Optimizing behavior through parameter adjustments and constraints

### Code Generation Examples

- [CodeLlama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)
- [Copilot](https://copilot.github.com/)
- [Cursor](https://cursor.com/)
- [Replit AI](https://replit.com/ai)
- [Q Developer](https://aws.amazon.com/q/developer/)
- [Cody](https://sourcegraph.com/docs)
- [Devin](https://www.cognition-labs.com/blog), the AI software engineer

## Peer programming with AI

- AI [features](https://www.cursor.com/features) of the [Cursor IDE](https://cursor.com/)
  - Code generation
  - Code editor
  - Rewriting and fixing code automatically
  - Enhanced predictions
  - Chat with your code
  - Link web documentation while chatting
- Setup
  - Download and install
    - May require some manual setup for Linux and MacOS
  - Create a free account
- Recommended settings
  - Tweak keyboard shortcuts
  - Import extensions and settings from VSCode
  - Privacy mode
  - Configuring models
- Usage
  - Fixing issues
  - Generating code
  - Chatting with your code
  - Using the chat
  - Generating commit messages
  - Generating documentation and test scripts based on other files

## Utilizing Sample Projects and Templates

- Leveraging existing resources
  - Starting from scratch vs. using templates or sample projects
    - Templates provide a foundation with basic structure and features implemented
    - Useful for learning, accelerating development, or building upon existing projects
  - Vercel's [template project list](https://vercel.com/templates)
    - Covers a wide range of use cases from simple landing pages to complex web applications
    - Valuable for learning, feature development, or project initialization
- Bootstrapping with `create-next-app`
  - [NextJS Boilerplate](https://vercel.com/templates/next.js/nextjs-boilerplate) initialization via `npx create-next-app`
    - Creates a new NextJS project with basic structure and features
    - Serves as a foundation for feature development or new projects
- [App Router Playground](https://app-router.vercel.app/): A showcase of `App Router` structure features
  - Provides working examples for implementation in projects
- Starter Projects
  - [Starter Projects](https://vercel.com/templates?type=starter) page offers diverse project bases
    - Covers various use cases from simple to complex applications
    - Designed for easy adaptation to specific project needs
  - Developers can create custom starter templates for future projects
- AI Templates
  - [AI Templates](https://vercel.com/templates?type=ai) page showcases AI model and API integration
    - Demonstrates Vercel AI and UI SDK usage in AI applications
    - Includes implementations from various AI providers
  - [AI Chat Template](https://chat.vercel.ai/): A comprehensive base for AI chat applications
    - Preconfigured with App Router, React Server Components, Suspense, and Server Actions
    - Supports multiple AI providers including OpenAI, Anthropic, Cohere, Hugging Face, and custom models
    - Tailwind CSS styling with `shadcn/ui` components
    - Serverless authentication via Vercel and [NextAuth.js](https://github.com/nextauthjs/next-auth)
    - Chat history, rate limiting, and session storage using [Vercel KV](https://vercel.com/storage/kv)

## Group Exercise

- Form a group of 2-5 people around you
- Create a Github repository for your group
- Create a landing page for your idea
- Commit the changes to the repository
- Deploy the application to Vercel
- Update your project submission with the link to your Vercel deployment
