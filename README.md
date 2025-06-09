---
title: Drift Detector
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
---

# Drift Detector
Drift Detector is an MCP server, designed to detect drift in LLM performance over time. 
This implementation is intended as a proof of concept and is not intended for production use.

## How to run

To run the Drift Detector, you need to have Python installed on your machine. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/saranshhalwai/drift-detector
    cd drift-detector
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the server:
    ```bash
   gradio app.py
    ```
4. Open your web browser and navigate to `http://localhost:7860` to access the Drift Detector interface.

## Interface

The interface consists of the following components:
- **Model Selection** - A panel allowing you to:
  - Select models from a dropdown list
  - Search for models by name or description
  - Create new models with custom system prompts
  - Enhance prompts with AI assistance

- **Model Operations** - A tabbed interface with:
  - **Chatbot** - Interact with the selected model through a conversational interface
  - **Drift Analysis** - Analyze and visualize model drift over time, including:
    - Calculate new drift scores for the selected model
    - View historical drift data in JSON format
    - Visualize drift trends through interactive charts

The drift detection functionality allows you to track changes in model performance over time, which is essential for monitoring and maintaining model quality.

## Under the Hood

Our GitHub repo consists of two main components:

- **Drift Detector Server**  
    A low-level MCP server that detects drift in LLM performance of the connected client.
- **Target Client**
    A client implemented using the fast-agent library, which connects to the Drift Detector server and demonstrates it's functionality.

The gradio interface in [app.py](app.py) is an example dashboard which allows users to interact with the Drift Detector server and visualize drift data.

### Drift Detector Server

The Drift Detector server is implemented using the MCP python SDK
    