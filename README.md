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
tags: 
- mcp-server-track
- agent-demo-track
---

This was made with the combined efforts of Saransh Halwai(HF username: [Sars6](https://huggingface.co/Sars6)), Harsh Bhati(HF username: [HarshBhati](https://huggingface.co/HarshBhati)), and Anurag Prasad(HF username: [LegendXInfinity](https://huggingface.co/LegendXInfinity))
GitHub repo: [Drift Detector](github.com/saranshhalwai/drift-detector)

# Drift Detector
Drift Detector is an MCP server, designed to detect drift in LLM performance over time by using the power of the **sampling** functionality of MCP. 
This implementation is intended as a **proof of concept** and is **NOT intended** for production use without significant changes.

## The Idea

The drift detector is a server that can be connected to any LLM client that supports the MCP sampling functionality. 
It allows you to monitor the performance of your LLM models over time, detecting any drift in their behavior.
This is particularly useful for applications where the model's performance may change due to various factors, such as changes in the data distribution, model updates, or other external influences.

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

### Database Integration

The system uses SQLite (by default) to store:
- Model information (name, capabilities, creation date)
- Drift history (date and score for each drift calculation)
- Diagnostic data (baseline and current questions/answers)

This enables persistent tracking of model performance over time, allowing for:
- Historical trend analysis
- Comparison between different models
- Early detection of performance degradation

### Drift Detector Server

The Drift Detector server is implemented using the MCP python SDK.
It exposes the following tools:

1. **run_initial_diagnostics**
   - **Purpose**: Establishes a baseline for model behavior using adaptive sampling techniques
   - **Parameters**:
     - `model`: The name of the model to run diagnostics on
     - `model_capabilities`: Full description of the model's capabilities and special features
   - **Sampling Process**:
     - First generates a tailored questionnaire based on model-specific capabilities
     - Collects responses by sampling the target model with controlled parameters (temperature=0.7)
     - Each question is processed individually to ensure proper context isolation
     - Baseline samples are stored as paired question-answer JSON records for future comparison
   - **Output**: Confirmation message indicating successful baseline creation

2. **check_drift**
   - **Purpose**: Measures potential drift by comparative sampling against the baseline
   - **Parameters**:
     - `model`: The name of the model to check for drift
   - **Sampling Process**:
     - Retrieves the original questions from the baseline
     - Re-samples the model with identical questions using the same sampling parameters
     - Maintains consistent context conditions to ensure fair comparison
     - Uses differential analysis to compare semantic and functional differences between sample sets
   - **Drift Evaluation**:
     - Calculates a numerical drift score based on answer divergence
     - Provides threshold-based alerts when drift exceeds acceptable limits (score > 50)
     - Stores the latest sample responses for audit and trend analysis

## Flow

The intended flow is as follows:
1. When the client contacts the server for the first time, it will run the `run_initial_diagnostics` tool.
2. The server will generate a tailored questionnaire based on the model's capabilities.
3. This questionnaire will be used to collect responses from the model, establishing a baseline for future comparisons.
4. Once the baseline is established, the server will store the paired question-answer JSON records in the database.
5. The client can then use the `check_drift` tool to measure potential drift in the model's performance.
6. The server will retrieve the original questions from the baseline and re-sample the model with identical questions.
7. The server will maintain consistent context conditions to ensure fair comparison.
8. If significant drift is detected (score > 50), the server will provide an alert and store the latest sample responses for audit and trend analysis.
9. The client can visualize the drift data through the Gradio interface, allowing users to track changes in model performance over time.

## Drift History Visualization

The system provides comprehensive visualization of drift history:

1. **Historical Data**: Real drift history is now fetched from the database rather than using mock data
2. **Interactive Charts**: Drift scores are plotted over time to identify trends
3. **Threshold Indicators**: Visual indicators show when drift exceeds acceptable limits
4. **Data Conversion**: Drift scores are normalized to percentages (0-100) for consistent display
5. **Error Handling**: Robust error handling for missing or malformed data

This real-time visualization allows users to:
- Identify gradual performance degradation
- Spot sudden changes in model behavior
- Make informed decisions about model retraining or replacement
- Compare drift patterns across different deployment environments

## Future Improvements

Potential enhancements for the Drift Detector include:
1. A full mcp server hosted over the cloud.
2. authentication and authorization for secure access.
1. Support for multiple database backends (PostgreSQL, MySQL)
2. Enhanced analytics and reporting features
3. Integration with CI/CD pipelines for automated monitoring
4. Advanced drift detection algorithms with explainability
5. Multi-metric drift analysis (beyond a single drift score)
6. User role-based access control for enterprise environments
