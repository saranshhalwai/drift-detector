import asyncio
import json
import os
from typing import Any

import mcp.types as types
from mcp import CreateMessageResult
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ourllm import genratequestionnaire, gradeanswers

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = Server("mcp-drift-server")

registered_models = {}

def get_all_models():
    """Retrieve all registered models."""
    return list(registered_models.keys())

def search_models(query: str):
    """Search registered models by name."""
    return [model for model in registered_models if query.lower() in model.lower()]

def get_model_details(model_name: str):
    """Get details of a specific model."""
    return registered_models.get(model_name, None)

def save_model(model_name: str, model_details: dict):
    """Save a new model or update an existing one."""
    registered_models[model_name] = model_details
    with open(os.path.join(DATA_DIR, "models.json"), "w") as f:
        json.dump(registered_models, f, indent=2)




@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_initial_diagnostics",
            description="Generate and store baseline diagnostics for a connected LLM.",
            inputSchema={"type":"object",
                         "properties": {
                              "model": {
                                  "type": "string",
                                  "description": "The name of the model to run diagnostics on"
                              },
                             "model_capabilities": {
                                    "type": "string",
                                    "description": "Full description of the model's capabilities, including any special features"
                             }
                         },

                          "required": ["model", "model_capabilities"]},
        ),
        types.Tool(
            name="check_drift",
            description="Re-run diagnostics and compare to baseline for drift scoring.",
            inputSchema={"type":"object",
                         "properties": {
                              "model": {
                                  "type": "string",
                                  "description": "The name of the model to run diagnostics on"
                              },
                         },

                          "required": ["model"]},
        ),
    ]


# === Sampling Wrapper ===
async def sample(messages: list[types.SamplingMessage], max_tokens=300) -> CreateMessageResult:
    return await app.request_context.session.create_message(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7
    )


# === Baseline File Paths ===
def get_baseline_path(model_name):
    return os.path.join(DATA_DIR, f"{model_name}_baseline.json")


def get_response_path(model_name):
    return os.path.join(DATA_DIR, f"{model_name}_latest.json")


# === Core Logic ===


async def run_initial_diagnostics(arguments: dict[str, Any]) -> list[types.TextContent]:
    if arguments and "model" in arguments:
        model = arguments["model"]
    else:
        raise(ValueError("Model details is required"))

    # 1. Ask the server's internal LLM to generate a questionnaire

    questions = await genratequestionnaire(model, arguments["model_capabilities"])  # Server-side trusted LLM

    # 2. Send questionnaire to target LLM (i.e., the client)
    answers = await sample(questions)  # Client model's answers

    # 3. Save Q/A pair
    with open(get_baseline_path(model), "w") as f:
        json.dump({
            "questions": [m.content.text for m in questions],
            "answers": [m.content.text for m in answers]
        }, f, indent=2)

    return [types.TextContent(type="text", text="Baseline stored for model: " + model)]



async def check_drift(arguments: dict[str, str]) -> list[types.TextContent]:
    if arguments and "model" in arguments:
        model = arguments["model"]
    else:
        raise (ValueError("Model details is required"))

    baseline_path = get_baseline_path(model)
    if not os.path.exists(baseline_path):
        return [types.TextContent(type="text", text="No baseline exists for model: " + model)]

    with open(baseline_path) as f:
        data = json.load(f)
        questions = [types.SamplingMessage(role="user", content=types.TextContent(type="text", text=q)) for q in
                     data["questions"]]
        old_answers = data["answers"]

    # 1. Ask the model again
    new_answers_msgs = await sample(questions)
    new_answers = [m.content.text for m in new_answers_msgs]


    grading_response = await gradeanswers(old_answers, new_answers)
    drift_score = grading_response[0].content.text.strip()

    # 3. Save the response
    with open(get_response_path(model), "w") as f:
        json.dump({
            "new_answers": new_answers,
            "drift_score": drift_score
        }, f, indent=2)

    # 4. Optionally alert if high drift
    alert = "🚨 Significant drift detected!" if float(drift_score) > 50 else "✅ Drift within acceptable limits."

    return [
        types.TextContent(type="text", text=f"Drift score for {model}: {drift_score}"),
        types.TextContent(type="text", text=alert)
    ]
@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None = None):
    if name == "run_initial_diagnostics":
        return await run_initial_diagnostics(arguments)
    elif name == "check_drift":
        return await check_drift(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

# === Entrypoint ===
async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
