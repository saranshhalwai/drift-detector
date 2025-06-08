# main.py
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


# Main entrypoint
async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
