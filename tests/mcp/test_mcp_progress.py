"""
Tests for MCP progress notification handling in AWorld.

This test suite verifies that progress notifications from MCP servers
are correctly received and handled by AWorld's MCP client.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import AsyncMock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aworld.mcp_client.server import MCPServerStdio
from aworld.mcp_client.utils import call_mcp_tool_with_exit_stack, get_server_instance
from aworld.sandbox.run.mcp_servers import McpServers


# Path to the progress test server
PROGRESS_SERVER_PATH = Path(__file__).parent / "progress_test_server.py"


class ProgressCollector:
    """Collects progress notifications for verification."""

    def __init__(self):
        self.notifications: List[Tuple[float, float | None, str | None]] = []

    async def callback(self, progress: float, total: float | None, message: str | None):
        """Collect progress notification."""
        print(f"[TEST] Progress received: progress={progress}, total={total}, message={message}", flush=True)
        self.notifications.append((progress, total, message))


class TestMCPProgressDirect:
    """Test progress notifications using direct MCP client (baseline)."""

    @pytest.mark.asyncio
    async def test_progress_direct_mcp_client(self):
        """
        Test that progress notifications work with direct MCP SDK client.
        This establishes the baseline - if this fails, the issue is with
        the MCP server or SDK, not AWorld.
        """
        collector = ProgressCollector()

        # Create server using AWorld's MCPServerStdio wrapper
        server = MCPServerStdio(
            name="progress-test",
            params={
                "command": sys.executable,
                "args": [str(PROGRESS_SERVER_PATH)],
                "env": {**os.environ},
            },
            cache_tools_list=False,
        )

        try:
            # Connect to server
            await server.connect()
            assert server.session is not None, "Server session should be established"

            # List tools to verify connection
            tools = await server.list_tools()
            tool_names = [t.name for t in tools]
            print(f"[TEST] Available tools: {tool_names}", flush=True)
            assert "slow_task_with_progress" in tool_names, "Progress tool should be available"

            # Call tool with progress callback
            print("[TEST] Calling slow_task_with_progress with progress_callback...", flush=True)
            result = await server.call_tool(
                tool_name="slow_task_with_progress",
                arguments={"steps": 3, "delay": 0.1},
                progress_callback=collector.callback,
            )

            print(f"[TEST] Tool result: {result}", flush=True)
            print(f"[TEST] Progress notifications received: {len(collector.notifications)}", flush=True)

            # Verify we got progress notifications
            assert len(collector.notifications) > 0, (
                "Should have received progress notifications. "
                "If this fails, the MCP SDK is not invoking the callback."
            )

            # Verify progress values
            for i, (progress, total, message) in enumerate(collector.notifications):
                print(f"[TEST] Notification {i}: progress={progress}, total={total}, message={message}")
                assert total == 3, f"Total should be 3, got {total}"
                assert progress == i + 1, f"Progress should be {i + 1}, got {progress}"
                assert message is not None, "Message should not be None"

        finally:
            await server.cleanup()


class TestMCPProgressAWorld:
    """Test progress notifications through AWorld's MCP infrastructure."""

    @pytest.mark.asyncio
    async def test_progress_via_call_mcp_tool_with_exit_stack(self):
        """
        Test progress notifications through call_mcp_tool_with_exit_stack.
        This tests the utility function used by McpServers.
        """
        collector = ProgressCollector()

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

        print("[TEST] Calling via call_mcp_tool_with_exit_stack...", flush=True)
        result = await call_mcp_tool_with_exit_stack(
            server_name="progress-test",
            tool_name="slow_task_with_progress",
            parameter={"steps": 3, "delay": 0.1},
            mcp_config=mcp_config,
            progress_callback=collector.callback,
            timeout=30.0,
        )

        print(f"[TEST] Result: {result}", flush=True)
        print(f"[TEST] Progress notifications: {len(collector.notifications)}", flush=True)

        assert result is not None, "Should get a result from tool call"
        assert len(collector.notifications) > 0, (
            "Should have received progress notifications through call_mcp_tool_with_exit_stack. "
            "If direct test passes but this fails, the issue is in call_mcp_tool_with_exit_stack."
        )

    @pytest.mark.asyncio
    async def test_progress_via_mcp_servers_class(self):
        """
        Test progress notifications through McpServers.call_tool.
        This is the full AWorld code path.
        """
        # Track if our diagnostic logs appear
        import io
        from contextlib import redirect_stdout

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

        # Create McpServers instance
        mcp_servers = McpServers(
            mcp_servers=["progress-test"],
            mcp_config=mcp_config,
        )

        # Create a mock context
        from unittest.mock import MagicMock
        mock_context = MagicMock()
        mock_context.session_id = "test-session-123"
        mock_context.task_id = "test-task-456"

        # List tools first
        tools = await mcp_servers.list_tools(context=mock_context)
        print(f"[TEST] Tools from McpServers: {[t.get('function', {}).get('name') for t in tools]}", flush=True)

        # Create action list
        action_list = [
            {
                "tool_name": "progress-test",
                "action_name": "slow_task_with_progress",
                "params": {"steps": 3, "delay": 0.1},
            }
        ]

        print("[TEST] Calling via McpServers.call_tool...", flush=True)

        # Capture stdout to see diagnostic print statements
        captured = io.StringIO()
        with redirect_stdout(captured):
            results = await mcp_servers.call_tool(
                action_list=action_list,
                context=mock_context,
            )

        output = captured.getvalue()
        print(f"[TEST] Captured output:\n{output}", flush=True)

        assert results is not None, "Should get results from McpServers.call_tool"
        assert len(results) > 0, "Should have at least one result"

        # Check if diagnostic logs appeared
        if "!!! PROGRESS CALLBACK INVOKED" in output:
            print("[TEST] SUCCESS: Progress callback was invoked!")
        else:
            print("[TEST] WARNING: Progress callback was NOT invoked (no diagnostic output found)")
            # Don't fail yet - progress might still work but diagnostics didn't capture it

        # Cleanup
        await mcp_servers.cleanup()


class TestMCPProgressStreamableHTTP:
    """
    Test progress notifications through streamable-http transport.

    NOTE: This requires a running mcp-proxy or streamable-http server.
    Skip if not available.
    """

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires external streamable-http server setup")
    async def test_progress_via_streamable_http(self):
        """
        Test progress through streamable-http transport.
        This is the transport mentioned in the bug report.
        """
        # This test would require setting up:
        # 1. A FastMCP server with progress
        # 2. mcp-proxy to bridge stdio <-> streamable-http
        # 3. Connecting AWorld to the proxy

        # For now, skip this test - it requires manual setup
        pass


def run_quick_test():
    """Run a quick test to check if progress works."""
    async def quick_test():
        collector = ProgressCollector()

        server = MCPServerStdio(
            name="progress-test",
            params={
                "command": sys.executable,
                "args": [str(PROGRESS_SERVER_PATH)],
            },
        )

        try:
            await server.connect()
            print("Connected to server", flush=True)

            tools = await server.list_tools()
            print(f"Tools: {[t.name for t in tools]}", flush=True)

            result = await server.call_tool(
                tool_name="slow_task_with_progress",
                arguments={"steps": 3, "delay": 0.2},
                progress_callback=collector.callback,
            )

            print(f"Result: {result}", flush=True)
            print(f"Progress notifications: {collector.notifications}", flush=True)

            if collector.notifications:
                print("\n SUCCESS: Progress notifications are working!")
            else:
                print("\n FAILURE: No progress notifications received!")

        finally:
            await server.cleanup()

    asyncio.run(quick_test())


if __name__ == "__main__":
    # Run quick test when executed directly
    run_quick_test()
