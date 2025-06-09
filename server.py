# server.py
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# Define diagnostic prompts statically for now
PROMPTS = {
    "drift-diagnostics": types.Prompt(
        name="drift-diagnostics",
        description="Run a diagnostic questionnaire to test LLM consistency.",
        arguments=[],
    )
}

# Setup server
app = Server("mcp-drift-server", version="0.1.0")


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return list(PROMPTS.values())


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
    if name not in PROMPTS:
        raise ValueError(f"Prompt not found: {name}")

    # Static message for MVP – replace with dynamic question set later
    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Answer the following: What's the capital of France?"
                )
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Explain why the sky is blue."
                )
            ),
        ]
    )

from mcp.server import Server
import mcp.types as types

# Assuming 'app' is your MCP Server instance

async def sample(app: Server, messages: list[types.SamplingMessage]):
    result = await app.request_context.session.create_message(
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return result

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="init_diagnostics",
            description="Run diagnostic questionnaire on the connected LLM.",
            inputSchema={"model_name": "Name of the LLM model"},
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, str] | None = None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Initializes diagnostics by running the questionnaire on the connected LLM.
    """
    # You could fetch dynamic questions here if needed
    questions = [
        types.SamplingMessage(role="user", content=types.TextContent(type="text", text="What is the capital of France?")),
        types.SamplingMessage(role="user", content=types.TextContent(type="text", text="Why is the sky blue?")),
    ]

    response = await sample(app, questions)

    # Return the assistant’s message(s) back to the caller
    return [types.TextContent(type="text", text=str(response.content))]

# Main entrypoint
async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
