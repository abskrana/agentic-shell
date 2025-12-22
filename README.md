# Agentic Shell

> An AI-powered terminal assistant that runs in your browser, combining a traditional command-line with intelligent execution modes, multi-language voice support, and a real-time chat interface.

Tell the agent what you want to do, and it will plan and execute commands for you, ask for clarification, or simply answer your questions.

## Features

-   **🌐 Browser-Based Interface:** A full-featured terminal and AI chat panel, side-by-side in your browser.
-   **🎯 4 Intelligent Execution Modes:**
    -   **Ask Mode:** Get instant answers and guidance without executing commands.
    -   **Task Mode:** The agent proposes a plan for your approval before running anything.
    -   **Auto Mode:** The agent plans and executes tasks autonomously.
    -   **Iterative Mode:** The agent executes one command at a time, analyzing the output to decide its next move.
-   **🎤 Multi-Language Voice Support:** Use your voice to give commands in 12 supported languages: English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, and Assamese.
-   **⚡ Real-Time Interaction:** Powered by WebSockets for instant command execution and output streaming.
-   **🧠 Smart Context:** The AI analyzes your terminal's screen to provide context-aware plans and answers.

## Project Architecture

This following diagram illustrates a hybrid architecture for the Agentic Shell project, separating the user-facing application from the heavy AI computation. The local application consists of a browser frontend for user interaction and a Python backend that manages a real shell process, orchestrating tasks in real-time via WebSockets.

<img width="3134" height="766" alt="image" src="https://github.com/user-attachments/assets/7b539bbf-f6db-4754-8ca1-02512e29c3cc" />

All intensive AI tasks, including model inference with Gemini/Qwen and speech-to-text, are offloaded to a scalable Lightning AI cloud backend through a unified API. This cloud backend also logs all interactions, which feeds into a distinct MLOps retraining pipeline that continuously fine-tunes and redeploys the models, creating a closed-loop system where the agent improves over time.

## Screenshot
<img width="1920" height="1020" alt="Screenshot 2025-11-11 224954" src="https://github.com/user-attachments/assets/f3bfbd64-a049-477a-b2b1-489d40f78635" />


## Installation

### 1. Prerequisites

-   Python 3.12 or higher.
-   `uv`: A fast Python package installer. Install it with:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
-   A **Lightning AI Backend URL** for AI and speech-to-text features.

### 2. Clone and Install

Clone the repository and run the interactive installation script.

```bash
git clone https://github.com/abhirana2109/agentic-shell.git
cd agentic-shell
./install.sh
```

The script will:
1.  Install all Python dependencies using `uv`.
2.  Prompt you to enter your **Lightning AI backend URL**.
3.  Create a `.env` file to store your configuration.
4.  Optionally, create a global `agsh` command to start the server from anywhere.

## How to Run the Agent

**If you created the global command during installation:**

Open a terminal and simply run:
```bash
agsh
```

**Otherwise, run from the project directory:**
```bash
./start.sh
```

Once the server is running, open your browser and navigate to **`http://localhost:8088`**.

## Using the Agent

1.  **Open the Web Interface:** Go to `http://localhost:8088`. You will see a terminal on the left and a chat panel on the right.
2.  **Select a Mode:** Use the mode toggle button at the bottom to switch between `Ask`, `Task`, `Auto`, and `Iterative` modes.
3.  **Choose a Model:** Switch between `Gemini` and `Qwen` AI models.
4.  **Type Your Goal:** Enter a natural language command (e.g., *"list all docker containers running"*) into the input bar.
5.  **Use Your Voice (Optional):**
    -   Select a language from the dropdown.
    -   Click the microphone icon to start and stop recording.
    -   Your speech will be transcribed into the input field.
6.  **Send:** Press Enter or click the send button to submit your request to the agent.

## Configuration

The application is configured via a `.env` file created by the `install.sh` script.

| Variable                | Description                                     | Default   |
| ----------------------- | ----------------------------------------------- | --------- |
| `LIGHTNING_UNIFIED_URL` | **Required.** Your Lightning AI backend URL.    | -         |
| `HOST`                  | The host address for the server.                | `0.0.0.0` |
| `PORT`                  | The port for the server.                        | `8088`    |

## Project Structure

The project is organized with a clear separation of concerns between the server, frontend, MLOps pipeline, and shared utilities.

-   **`agentic-shell/` (Root Directory)**
    -   **`main.py`**: The main entry point that launches the web server.
    -   **`server.py`**: Initializes the `aiohttp` web app, Socket.IO server, and the core `bash` pseudo-terminal (PTY) process that powers the interactive shell.
    -   **`config.py`**: A centralized configuration hub. It defines server settings, execution timeouts, supported languages, and contains all the prompt templates used to instruct the AI models.
    -   **`ai_brain.py`**: The core AI logic module. It interfaces with the AI models to generate execution plans, make decisions in iterative mode, and formulate natural language answers.
    -   **`socket_handlers.py`**: Manages all real-time communication. It defines WebSocket event listeners for user input, prompt requests, plan approvals, and voice transcription.
    -   **`mode_handlers.py`**: Contains the specific logic for each of the four execution modes (Ask, Task, Auto, Iterative), orchestrating the interaction between the AI brain and the terminal.
    -   **`lightning_client.py`**: A dedicated client for communicating with the Lightning AI cloud backend. It handles all outgoing API requests for AI generation and speech-to-text.
    -   **`install.sh`**: An interactive installation script that sets up the Python environment, dependencies, and the `.env` configuration file.
    -   **`start.sh`**: A helper script that activates the virtual environment and starts the web server.
    -   **`agsh`**: A global launcher script that allows the user to start the server from any directory in their terminal.
    -   **`pyproject.toml`**: Defines project metadata and dependencies according to modern Python packaging standards (PEP 621).
    -   **`requirements.txt`**: A standard list of Python dependencies, primarily for compatibility and reference.
    -   **`uv.lock`**: A lock file generated by `uv` to ensure deterministic and reproducible installation of all dependencies.

-   **`static/` (Frontend)**
    -   **`index.html`**: The single HTML file that defines the structure of the web page, including the terminal and chat panels.
    -   **`main.js`**: The heart of the frontend. It manages the terminal, Socket.IO connection, voice input, and all user interactions in the browser.
    -   **`style.css`**: Custom CSS to style the layout, chat messages, control bar, and overall theme of the web interface.

-   **`utils/` (Shared Helpers)**
    -   **`__init__.py`**: Makes the `utils` directory a Python package, enabling easier module imports.
    -   **`logger.py`**: Manages structured logging and sends interaction data to the cloud backend for analysis and retraining.
    -   **`messaging.py`**: Provides a simple helper function for sending formatted agent messages over WebSockets.
    -   **`terminal_utils.py`**: A key utility for analyzing terminal output to intelligently detect when commands have finished executing.

-   **`LightningAI/` (Cloud Backend & MLOps Pipeline)**
    -   **`lightning_unified_backend.py`**: A single FastAPI application that serves as the project's external brain. It exposes endpoints for: LLM inference (routing to Gemini or Qwen), speech-to-text transcription, and receiving interaction logs for MLOps.
    -   **`qwen_adapter.py`**: A compatibility layer that allows the backend to interact with a vLLM-hosted Qwen model using a Gemini-like interface, ensuring model interchangeability.
    -   **`train_bash_agent.py`**: A script for the *initial* fine-tuning of the base Qwen model on a public dataset (`nl2bash`) to give it a strong foundation in command generation.
    -   **`retraining.py`**: The core script for the continuous MLOps loop. It takes recent user interaction logs, formats them into a new dataset, and further fine-tunes the model.
    -   **`merge_and_save.py`**: A utility script that merges the fine-tuned LoRA adapter (the result of retraining) back into the base model to create a new, standalone model.
    -   **`deploy_api.py`**: A script to serve the newly merged model using the vLLM OpenAI-compatible API server, making it ready for inference.
    -   **`pipeline.py`**: The master script that orchestrates the entire MLOps workflow, automatically running the `retraining`, `merge_and_save`, and `deploy_api` steps in sequence.
