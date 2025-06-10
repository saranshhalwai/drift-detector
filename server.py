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
                    "model_capabilities": {"type": "string",
                                           "description": "Full description of the model's capabilities, along with the system prompt."}
                },
                "required": ["model", "model_capabilities"]
            },
        ),
        types.Tool(
            name="check_drift",
            description="Re-run diagnostics and compare to baseline for drift scoring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "The name of the model to run diagnostics on"}},
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
    try:
        return await app.request_context.session.create_message(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
    except Exception as e:
        print(f"Error in sampling: {e}")
        # Return a fallback response
        return CreateMessageResult(
            content=types.TextContent(type="text", text="Error generating response"),
            model="unknown",
            role="assistant"
        )


# === Core Logic ===
async def run_initial_diagnostics(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model = arguments["model"]
    capabilities = arguments["model_capabilities"]

    try:
        # 1. Generate questionnaire using ourllm (returns list of strings)
        questions = genratequestionnaire(model, capabilities)

        # 2. Convert questions to sampling messages and get answers
        answers = []
        for question_text in questions:
            try:
                sampling_msg = types.SamplingMessage(
                    role="user",
                    content=types.TextContent(type="text", text=question_text)
                )
                answer_result = await sample([sampling_msg])

                # Extract text content from the answer
                if hasattr(answer_result, 'content'):
                    if hasattr(answer_result.content, 'text'):
                        answers.append(answer_result.content.text)
                    else:
                        answers.append(str(answer_result.content))
                else:
                    answers.append("No response generated")

            except Exception as e:
                print(f"Error getting answer for question '{question_text}': {e}")
                answers.append(f"Error: {str(e)}")

        # 3. Save the model capabilities and questions/answers to database
        try:
            register_model_with_capabilities(model, capabilities)
            save_diagnostic_data(
                model_name=model,
                questions=questions,
                answers=answers,
                is_baseline=True
            )
        except Exception as e:
            print(f"Error saving diagnostic data: {e}")
            return [types.TextContent(type="text", text=f"❌ Error saving baseline for {model}: {str(e)}")]

        return [
            types.TextContent(type="text", text=f"✅ Baseline stored for model: {model} ({len(questions)} questions)")]

    except Exception as e:
        print(f"Error in run_initial_diagnostics: {e}")
        return [types.TextContent(type="text", text=f"❌ Error running diagnostics for {model}: {str(e)}")]


async def check_drift(arguments: Dict[str, Any]) -> List[types.TextContent]:
    model = arguments["model"]

    try:
        # Get baseline from database
        baseline = get_baseline_diagnostics(model)

        # Ensure baseline exists
        if not baseline:
            return [types.TextContent(type="text", text=f"❌ No baseline for model: {model}")]

        # Get old answers from baseline
        old_answers = baseline["answers"]
        questions = baseline["questions"]

        # Ask the model the same questions again
        new_answers = []
        for question_text in questions:
            try:
                sampling_msg = types.SamplingMessage(
                    role="user",
                    content=types.TextContent(type="text", text=question_text)
                )
                answer_result = await sample([sampling_msg])

                # Extract text content from the answer
                if hasattr(answer_result, 'content'):
                    if hasattr(answer_result.content, 'text'):
                        new_answers.append(answer_result.content.text)
                    else:
                        new_answers.append(str(answer_result.content))
                else:
                    new_answers.append("No response generated")

            except Exception as e:
                print(f"Error getting new answer for question '{question_text}': {e}")
                new_answers.append(f"Error: {str(e)}")

        # Grade the answers and get a drift score (returns string)
        drift_score_str = gradeanswers(old_answers, new_answers)

        # Save the latest responses and drift score to database
        try:
            save_diagnostic_data(
                model_name=model,
                questions=questions,
                answers=new_answers,
                is_baseline=False
            )
            save_drift_score(model, drift_score_str)
        except Exception as e:
            print(f"Error saving drift data: {e}")

        # Alert threshold
        try:
            score_val = float(drift_score_str)
            alert = "🚨 Significant drift!" if score_val > 50 else "✅ Drift OK"
        except ValueError:
            alert = "⚠️ Drift score not numeric"

        return [
            types.TextContent(type="text", text=f"Drift score for {model}: {drift_score_str}%"),
            types.TextContent(type="text", text=alert)
        ]

    except Exception as e:
        print(f"Error in check_drift: {e}")
        return [types.TextContent(type="text", text=f"❌ Error checking drift for {model}: {str(e)}")]


# Database tool handlers
async def get_all_models_handler_async(_: Dict[str, Any]) -> List[types.TextContent]:
    try:
        models = get_all_models_handler({})
        if not models:
            return [types.TextContent(type="text", text="No models registered.")]

        model_list = "\n".join([f"• {m['name']} - {m.get('description', 'No description')}" for m in models])
        return [types.TextContent(
            type="text",
            text=f"Registered models:\n{model_list}"
        )]
    except Exception as e:
        print(f"Error getting all models: {e}")
        return [types.TextContent(type="text", text=f"❌ Error retrieving models: {str(e)}")]


async def search_models_handler_async(arguments: Dict[str, Any]) -> List[types.TextContent]:
    try:
        query = arguments.get("query", "")
        models = search_models_handler({"search_term": query})

        if not models:
            return [types.TextContent(
                type="text",
                text=f"No models found matching '{query}'."
            )]

        model_list = "\n".join([f"• {m['name']} - {m.get('description', 'No description')}" for m in models])
        return [types.TextContent(
            type="text",
            text=f"Models matching '{query}':\n{model_list}"
        )]
    except Exception as e:
        print(f"Error searching models: {e}")
        return [types.TextContent(type="text", text=f"❌ Error searching models: {str(e)}")]


# === Dispatcher ===
@app.call_tool()
async def dispatch_tool(name: str, arguments: Dict[str, Any] | None = None):
    try:
        if name == "run_initial_diagnostics":
            return await run_initial_diagnostics(arguments)
        elif name == "check_drift":
            return await check_drift(arguments)
        elif name == "get_all_models":
            return await get_all_models_handler_async(arguments or {})
        elif name == "search_models":
            return await search_models_handler_async(arguments or {})
        else:
            return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    except Exception as e:
        print(f"Error in dispatch_tool for {name}: {e}")
        return [types.TextContent(type="text", text=f"❌ Error executing {name}: {str(e)}")]


# === Entrypoint ===
async def main():
    try:
        async with stdio_server() as (reader, writer):
            await app.run(reader, writer, app.create_initialization_options())
    except Exception as e:
        print(f"Error running MCP server: {e}")


if __name__ == "__main__":
    asyncio.run(main())