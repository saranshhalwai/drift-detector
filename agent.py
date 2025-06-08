# agent.py
import asyncio
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt

fast = FastAgent("Drift Test Agent")

@fast.agent(
    name="diagnostics",
    instruction="Answer diagnostic questions to test LLM stability.",
    servers=["drift-server"]
)
async def main():
    async with fast.run() as agent:
        # Apply prompt from the MCP server
        print(">> Getting prompt from MCP server…")
        result = await agent.apply_prompt("drift-diagnostics")
        print(">> Response:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
