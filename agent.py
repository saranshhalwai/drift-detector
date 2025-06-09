import asyncio
from mcp_agent.core.fastagent import FastAgent

fast = FastAgent("Drift Test Agent")


@fast.agent(
    name="diagnostics",
    instruction="Your name is 'diagnostics'. Run diagnostics using the MCP server tool.",
    servers=["drift-server"]
)
async def main():
    async with fast.run() as agent:
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
