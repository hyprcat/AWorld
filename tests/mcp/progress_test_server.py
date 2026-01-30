"""
Simple FastMCP server that sends progress notifications for testing.

This server provides a tool that reports progress during execution,
used to test if AWorld correctly receives MCP progress notifications.
"""

import asyncio
import sys
from mcp.server.fastmcp import FastMCP, Context

# Create the FastMCP server
mcp = FastMCP("progress-test-server")


@mcp.tool(description="A test tool that sends progress notifications during execution")
async def slow_task_with_progress(
    steps: int = 5,
    delay: float = 0.5,
    ctx: Context = None,
) -> str:
    """
    Execute a slow task that reports progress at each step.

    Args:
        steps: Number of steps to execute (default: 5)
        delay: Delay in seconds between steps (default: 0.5)
        ctx: MCP context for reporting progress

    Returns:
        Completion message with step count
    """
    print(f"[SERVER] Starting slow_task_with_progress with {steps} steps", file=sys.stderr, flush=True)

    for i in range(steps):
        # Report progress
        if ctx:
            progress = (i + 1) / steps * 100
            message = f"Processing step {i + 1} of {steps}"
            print(f"[SERVER] Reporting progress: {progress}% - {message}", file=sys.stderr, flush=True)
            await ctx.report_progress(
                progress=i + 1,
                total=steps,
                message=message
            )

        # Simulate work
        await asyncio.sleep(delay)

    result = f"Completed {steps} steps successfully"
    print(f"[SERVER] Task completed: {result}", file=sys.stderr, flush=True)
    return result


@mcp.tool(description="A simple tool without progress for comparison")
def simple_task() -> str:
    """A simple tool that returns immediately without progress."""
    return "Simple task completed"


def main():
    """Run the MCP server."""
    print("[SERVER] Starting progress-test-server...", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
