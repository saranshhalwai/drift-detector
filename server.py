import asyncio
import os
from typing import Any, List, Dict

import mcp.types as types
from mcp import CreateMessageResult
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ourllm import genratequestionnaire, gradeanswers
from database_module import init_db
from database_module import (
    get_all_models_handler,
    search_models_handler,
    save_diagnostic_data,
    get_baseline_diagnostics,
    save_drift_score,
    register_model_with_capabilities
)

# Initialize data directory and database
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
init_db()

app = Server("mcp-drift-server")

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
                    "model_capabilities": {"type": "string", "description": "Full description of the model's capabilities, along with the system prompt."}
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

# === Core Logic ===
async def run_initial_diagnostics(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model = arguments["model"]
    capabilities = arguments["model_capabilities"]

    # 1. Ask the server's internal LLM to generate a questionnaire
    questions = genratequestionnaire(model, capabilities)  # Server-side trusted LLM
    answers = []
    for q in questions:
        a = await sample([q])
        answers.append(a)

    # 2. Save the model capabilities and questions/answers to database
    register_model_with_capabilities(model, capabilities)
    save_diagnostic_data(
        model_name=model,
        questions=[m.content.text for m in questions],
        answers=[m.content.text for m in answers],
        is_baseline=True
    )

    return [types.TextContent(type="text", text=f"✅ Baseline stored for model: {model}")]

async def check_drift(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model = arguments["model"]

    # Get baseline from database
    baseline = get_baseline_diagnostics(model)

    # Ensure baseline exists
    if not baseline:
        return [types.TextContent(type="text", text=f"❌ No baseline for model: {model}")]

    # Convert questions to sampling messages
    questions = [
        types.SamplingMessage(role="user", content=types.TextContent(type="text", text=q))
        for q in baseline["questions"]
    ]
    old_answers = baseline["answers"]

    # Ask the model again
    new_answers_msgs = []
    for q in questions:
        a = await sample([q])
        new_answers_msgs.append(a)
    new_answers = [m.content.text for m in new_answers_msgs]

    # Grade the answers and get a drift score
    grading_response = gradeanswers(old_answers, new_answers)
    drift_score = grading_response[0].content.text.strip()

    # Save the latest responses and drift score to database
    save_diagnostic_data(
        model_name=model,
        questions=baseline["questions"],
        answers=new_answers,
        is_baseline=False
    )
    save_drift_score(model, drift_score)

    # Alert threshold
    try:
        score_val = float(drift_score)
        alert = "🚨 Significant drift!" if score_val > 50 else "✅ Drift OK"
    except ValueError:
        alert = "⚠️ Drift score not numeric"

    return [
        types.TextContent(type="text", text=f"Drift score for {model}: {drift_score}"),
        types.TextContent(type="text", text=alert)
    ]

# Database tool handlers
async def get_all_models_handler_async(_: Dict[str, Any]) -> List[types.TextContent]:
    models = get_all_models_handler({})
    if not models:
        return [types.TextContent(type="text", text="No models registered.")]

    model_list = "\n".join([f"• {m['name']} - {m['description']}" for m in models])
    return [types.TextContent(
        type="text",
        text=f"Registered models:\n{model_list}"
    )]

async def search_models_handler_async(arguments: Dict[str, Any]) -> List[types.TextContent]:
    query = arguments.get("query", "")
    models = search_models_handler({"search_term": query})

    if not models:
        return [types.TextContent(
            type="text",
            text=f"No models found matching '{query}'."
        )]

    model_list = "\n".join([f"• {m['name']} - {m['description']}" for m in models])
    return [types.TextContent(
        type="text",
        text=f"Models matching '{query}':\n{model_list}"
    )]

# === Dispatcher ===
@app.call_tool()
async def dispatch_tool(name: str, arguments: Dict[str, Any] | None = None):
    if name == "run_initial_diagnostics":
        return await run_initial_diagnostics(arguments)
    elif name == "check_drift":
        return await check_drift(arguments)
    elif name == "get_all_models":
        return await get_all_models_handler_async(arguments or {})
    elif name == "search_models":
        return await search_models_handler_async(arguments or {})
    else:
        raise ValueError(f"Unknown tool: {name}")

# === Entrypoint ===
async def main():
    async with stdio_server() as (reader, writer):
        await app.run(reader, writer, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
