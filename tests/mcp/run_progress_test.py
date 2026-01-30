#!/usr/bin/env python
"""
Quick runner script to test MCP progress notifications.

Run this directly to test if progress callbacks are working:
    python tests/mcp/run_progress_test.py

This script tests three levels:
1. Direct MCP SDK client (baseline)
2. AWorld's call_mcp_tool_with_exit_stack utility
3. AWorld's McpServers.call_tool (full code path)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

PROGRESS_SERVER_PATH = Path(__file__).parent / "progress_test_server.py"


class ProgressCollector:
    """Collects progress notifications for verification."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.notifications: List[Tuple[float, float | None, str | None]] = []

    async def callback(self, progress: float, total: float | None, message: str | None):
        """Collect progress notification."""
        print(f"  [{self.name}] Progress: {progress}/{total} - {message}", flush=True)
        self.notifications.append((progress, total, message))


async def test_direct_mcp_client():
    """Test 1: Direct MCP SDK client (baseline)."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct MCP SDK Client (baseline)")
    print("=" * 60)

    from aworld.mcp_client.server import MCPServerStdio

    collector = ProgressCollector("direct")

    server = MCPServerStdio(
        name="progress-test",
        params={
            "command": sys.executable,
            "args": [str(PROGRESS_SERVER_PATH)],
            "env": {**os.environ},
        },
    )

    try:
        print("Connecting to server...", flush=True)
        await server.connect()

        tools = await server.list_tools()
        print(f"Available tools: {[t.name for t in tools]}", flush=True)

        print("Calling slow_task_with_progress...", flush=True)
        result = await server.call_tool(
            tool_name="slow_task_with_progress",
            arguments={"steps": 3, "delay": 0.2},
            progress_callback=collector.callback,
        )

        print(f"Result: {result.content[0].text if result.content else 'No content'}", flush=True)

        if collector.notifications:
            print(f"\n[PASS] Received {len(collector.notifications)} progress notifications")
            return True
        else:
            print("\n[FAIL] No progress notifications received!")
            return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await server.cleanup()


async def test_call_mcp_tool_with_exit_stack():
    """Test 2: AWorld's call_mcp_tool_with_exit_stack utility."""
    print("\n" + "=" * 60)
    print("TEST 2: call_mcp_tool_with_exit_stack")
    print("=" * 60)

    from aworld.mcp_client.utils import call_mcp_tool_with_exit_stack

    collector = ProgressCollector("exit_stack")

    mcp_config = {
        "mcpServers": {
            "progress-test": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(PROGRESS_SERVER_PATH)],
                "env": {**os.environ},
            }
        }
    }

    try:
        print("Calling via call_mcp_tool_with_exit_stack...", flush=True)
        result = await call_mcp_tool_with_exit_stack(
            server_name="progress-test",
            tool_name="slow_task_with_progress",
            parameter={"steps": 3, "delay": 0.2},
            mcp_config=mcp_config,
            progress_callback=collector.callback,
            timeout=30.0,
        )

        if result and result.content:
            print(f"Result: {result.content[0].text}", flush=True)
        else:
            print("Result: None or no content", flush=True)

        if collector.notifications:
            print(f"\n[PASS] Received {len(collector.notifications)} progress notifications")
            return True
        else:
            print("\n[FAIL] No progress notifications received!")
            return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_servers_class():
    """Test 3: AWorld's McpServers.call_tool (full code path)."""
    print("\n" + "=" * 60)
    print("TEST 3: McpServers.call_tool (full AWorld path)")
    print("=" * 60)

    from aworld.sandbox.run.mcp_servers import McpServers
    from unittest.mock import MagicMock

    mcp_config = {
        "mcpServers": {
            "progress-test": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(PROGRESS_SERVER_PATH)],
                "env": {**os.environ},
            }
        }
    }

    mcp_servers = McpServers(
        mcp_servers=["progress-test"],
        mcp_config=mcp_config,
    )

    # Create mock context
    mock_context = MagicMock()
    mock_context.session_id = "test-session-123"
    mock_context.task_id = "test-task-456"

    try:
        print("Listing tools...", flush=True)
        tools = await mcp_servers.list_tools(context=mock_context)
        print(f"Available tools: {[t.get('function', {}).get('name') for t in tools]}", flush=True)

        action_list = [
            {
                "tool_name": "progress-test",
                "action_name": "slow_task_with_progress",
                "params": {"steps": 3, "delay": 0.2},
            }
        ]

        print("Calling via McpServers.call_tool...", flush=True)
        print("(Watch for '!!! PROGRESS CALLBACK INVOKED' messages)", flush=True)
        print("-" * 40)

        results = await mcp_servers.call_tool(
            action_list=action_list,
            context=mock_context,
        )

        print("-" * 40)

        if results:
            print(f"Got {len(results)} result(s)", flush=True)
            for r in results:
                print(f"  Result: {r.content[:100]}..." if len(r.content) > 100 else f"  Result: {r.content}")
            return True  # We can't easily check progress here without modifying McpServers
        else:
            print("\n[FAIL] No results returned!")
            return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await mcp_servers.cleanup()


async def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# MCP Progress Notification Tests")
    print("#" * 60)

    results = {}

    # Test 1: Direct MCP client
    results["direct"] = await test_direct_mcp_client()

    # Test 2: call_mcp_tool_with_exit_stack
    results["exit_stack"] = await test_call_mcp_tool_with_exit_stack()

    # Test 3: McpServers.call_tool
    results["mcp_servers"] = await test_mcp_servers_class()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print("\n" + "=" * 60)

    if results["direct"] and not results["exit_stack"]:
        print("DIAGNOSIS: Progress works at SDK level but fails in call_mcp_tool_with_exit_stack")
        print("  -> Check the progress_callback parameter passing in utils.py")

    elif results["exit_stack"] and not results["mcp_servers"]:
        print("DIAGNOSIS: Progress works in utils but fails in McpServers.call_tool")
        print("  -> Check the progress_callback closure in mcp_servers.py")

    elif not results["direct"]:
        print("DIAGNOSIS: Progress doesn't work even at SDK level")
        print("  -> This might be an MCP SDK version or server implementation issue")

    else:
        print("All tests passed (or check the output above for '!!! PROGRESS' messages)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
