import asyncio
import json
import os
from typing import Any, List, Dict

import mcp.types as types
from mcp import CreateMessageResult
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ourllm import genratequestionnaire, gradeanswers
from database_module import init_db
from database_module import get_all_models_handler, search_models_handler

# Initialize data directory and database
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
init_db()

app = Server("mcp-drift-server")


# === Sampling Helper ===
async def sample(messages: List[types.SamplingMessage], max_tokens: int = 300) -> CreateMessageResult:
    return await app.request_context.session.create_message(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )


# === Baseline File Helpers ===
def get_baseline_path(model_name: str) -> str:
    return os.path.join(DATA_DIR, f"{model_name}_baseline.json")


def get_response_path(model_name: str) -> str:
    return os.path.join(DATA_DIR, f"{model_name}_latest.json")


# === Tool Manifest ===
@app.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="run_initial_diagnostics",
            description="Generate and store baseline diagnostics for a connected LLM.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The name of the model to run diagnostics on"},
                    "model_capabilities": {"type": "string", "description": "Full description of the model's capabilities"}
                },
                "required": ["model", "model_capabilities"]
            },
        ),
        types.Tool(
            name="check_drift",
            description="Re-run diagnostics and compare to baseline for drift scoring.",
            inputSchema={
                "type": "object",
                "properties": {"model": {"type": "string", "description": "The name of the model to run diagnostics on"}},
                "required": ["model"]
            },
        ),
        types.Tool(
            name="get_all_models",
            description="Retrieve all registered models from the database.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="search_models",
            description="Search registered models by name.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Substring to match model names against"}},
                "required": ["query"]
            }
        ),
    ]



# === Sampling Wrapper ===
async def sample(messages: list[types.SamplingMessage], max_tokens=600) -> CreateMessageResult:
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
async def run_initial_diagnostics(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model = arguments["model"]
    caps  = arguments["model_capabilities"]

    # 1. Generate questionnaire
    questions = await genratequestionnaire(model, caps)

    # 2. Ask the target LLM (client)
    answers = await sample(questions)


    # 3. Persist baseline

    # 1. Ask the server's internal LLM to generate a questionnaire

    questions = genratequestionnaire(model, arguments["model_capabilities"])  # Server-side trusted LLM
    answers = []
    for q in questions:
        a = await sample([q])
        answers.append(a)

    # 3. Save Q/A pair

    with open(get_baseline_path(model), "w") as f:
        json.dump({
            "questions": [m.content.text for m in questions],
            "answers":   [m.content.text for m in answers]
        }, f, indent=2)

    return [types.TextContent(type="text", text=f"✅ Baseline stored for model: {model}")]


async def check_drift(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model     = arguments["model"]
    base_path = get_baseline_path(model)

    # Ensure baseline exists
    if not os.path.exists(base_path):
        return [types.TextContent(type="text", text=f"❌ No baseline for model: {model}")]

    # Load questions + old answers
    with open(base_path) as f:
        data = json.load(f)
    questions   = [
        types.SamplingMessage(role="user", content=types.TextContent(type="text", text=q))
        for q in data["questions"]
    ]
    old_answers = data["answers"]


    # 1. Get fresh answers
    new_msgs    = await sample(questions)
    new_answers = [m.content.text for m in new_msgs]

    # 1. Ask the model again
    new_answers_msgs = []
    for q in questions:
        a = await sample([q])
        new_answers_msgs.append(a)
    new_answers = [m.content.text for m in new_answers_msgs]


    # 2. Grade for drift
    grading     = await gradeanswers(old_answers, new_answers)
    drift_score = grading[0].content.text.strip()


    # 3. Save latest
    grading_response = gradeanswers(old_answers, new_answers)
    drift_score = grading_response[0].content.text.strip()

    # 3. Save the response

    with open(get_response_path(model), "w") as f:
        json.dump({
            "new_answers": new_answers,
            "drift_score": drift_score
        }, f, indent=2)

    # 4. Alert threshold
    try:
        score_val = float(drift_score)
        alert     = "🚨 Significant drift!" if score_val > 50 else "✅ Drift OK"
    except ValueError:
        alert = "⚠️ Drift score not numeric"

    return [
        types.TextContent(type="text", text=f"Drift score for {model}: {drift_score}"),
        types.TextContent(type="text", text=alert)
    ]


# === Dispatcher ===
@app.call_tool()
async def dispatch_tool(name: str, arguments: Dict[str, Any] | None = None):
    if name == "run_initial_diagnostics":
        return await run_initial_diagnostics(arguments or {})
    if name == "check_drift":
        return await check_drift(arguments or {})
    if name == "get_all_models":
        return await get_all_models_handler()
    if name == "search_models":
        return await search_models_handler(arguments or {})
    raise ValueError(f"Unknown tool: {name}")


# === Entrypoint ===
async def main():
    async with stdio_server() as (reader, writer):
        await app.run(reader, writer, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
