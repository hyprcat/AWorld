"""
Parallel MCP server loader - speeds up tool loading by connecting to servers concurrently.

This module provides a drop-in replacement for the sequential server loading in utils.py.
It patches mcp_tool_desc_transform_v2 to load MCP servers in parallel instead of sequentially.

Usage:
    # Option 1: Enable via environment variable (recommended)
    # Set AWORLD_PARALLEL_MCP=1 before importing aworld

    # Option 2: Explicit patching
    from aworld.mcp_client.parallel_loader import enable_parallel_loading
    enable_parallel_loading()

    # Option 3: Use the parallel function directly
    from aworld.mcp_client.parallel_loader import mcp_tool_desc_transform_v2_parallel
"""

import asyncio
import os
import traceback
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import List, Dict, Any, Optional

from aworld.logs.util import logger

# Store reference to original function for potential restoration
_original_mcp_tool_desc_transform_v2 = None


async def _load_single_mcp_server(
    server_config: Dict[str, Any],
    context: Any,
    sandbox_id: Optional[str],
    black_tool_actions: Dict[str, List[str]],
    tool_actions: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """
    Load tools from a single MCP server.

    This is extracted to allow parallel execution via asyncio.gather().
    """
    from aworld.mcp_client.server import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp
    from aworld.mcp_client.utils import run

    try:
        async with AsyncExitStack() as stack:
            if server_config["type"] == "sse":
                params = server_config["params"].copy()
                headers = params.get("headers") or {}
                env_name = headers.get("env_name")
                _SESSION_ID = env_name or ""
                if sandbox_id:
                    _SESSION_ID = _SESSION_ID + "_" + sandbox_id if _SESSION_ID else sandbox_id
                    from aworld.core.context.amni import AmniContext
                    if isinstance(context, AmniContext) and context.get_config().env_config.isolate:
                        if context.task_id:
                            _SESSION_ID = _SESSION_ID + "_" + str(context.task_id)
                    headers["SESSION_ID"] = _SESSION_ID

                params["headers"] = headers
                server = MCPServerSse(
                    name=server_config["name"], params=params
                )
            elif server_config["type"] == "streamable-http":
                params = server_config["params"].copy()
                headers = params.get("headers") or {}
                env_name = headers.get("env_name")
                _SESSION_ID = env_name or ""
                if sandbox_id:
                    _SESSION_ID = _SESSION_ID + "_" + sandbox_id if _SESSION_ID else sandbox_id
                    from aworld.core.context.amni import AmniContext
                    if isinstance(context, AmniContext) and context.get_config().env_config.isolate:
                        if context.task_id:
                            _SESSION_ID = _SESSION_ID + "_" + str(context.task_id)
                    headers["SESSION_ID"] = _SESSION_ID

                params["headers"] = headers
                if "timeout" in params and not isinstance(params["timeout"], timedelta):
                    params["timeout"] = timedelta(seconds=float(params["timeout"]))
                if "sse_read_timeout" in params and not isinstance(params["sse_read_timeout"], timedelta):
                    params["sse_read_timeout"] = timedelta(seconds=float(params["sse_read_timeout"]))
                server = MCPServerStreamableHttp(
                    name=server_config["name"], params=params
                )
            elif server_config["type"] == "stdio":
                server = MCPServerStdio(
                    name=server_config["name"], params=server_config["params"]
                )
            else:
                logger.warning(
                    f"Unsupported MCP server type: {server_config['type']}"
                )
                return []

            server = await stack.enter_async_context(server)
            return await run(
                mcp_servers=[server],
                black_tool_actions=black_tool_actions,
                tool_actions=tool_actions
            )
    except BaseException as err:
        logger.warning(
            f"Failed to get tools for MCP server '{server_config['name']}'.\n"
            f"Error: {err}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return []


async def mcp_tool_desc_transform_v2_parallel(
    tools: List[str] = None,
    mcp_config: Dict[str, Any] = None,
    context: Any = None,
    server_instances: Dict[str, Any] = None,
    black_tool_actions: Dict[str, List[str]] = None,
    sandbox_id: Optional[str] = None,
    tool_actions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Parallel version of mcp_tool_desc_transform_v2.

    Loads all MCP servers concurrently using asyncio.gather() instead of
    sequentially, significantly reducing total loading time when multiple
    servers are configured.
    """
    import json
    import requests
    from aworld.mcp_client.utils import get_function_tool, MCP_SERVERS_CONFIG
    import aworld.mcp_client.utils as utils_module

    if not mcp_config:
        return []

    config = mcp_config
    # Update global config
    utils_module.MCP_SERVERS_CONFIG = config

    mcp_servers_config = config.get("mcpServers", {})
    server_configs = []
    openai_tools = []

    # First pass: handle non-MCP servers (function_tool, api) and collect MCP server configs
    for server_name, server_config in mcp_servers_config.items():
        if server_config.get("disabled", False):
            continue

        if tools and server_name in tools:
            server_type = server_config.get("type", "")

            if server_type == "function_tool":
                try:
                    tmp_function_tool = get_function_tool(server_name)
                    openai_tools.extend(tmp_function_tool)
                except Exception as e:
                    logger.warning(f"server_name:{server_name} translate failed: {e}")

            elif server_type == "api":
                try:
                    api_result = requests.get(server_config["url"] + "/list_tools")
                    if api_result and api_result.text:
                        data = json.loads(api_result.text)
                        if data and data.get("tools"):
                            for item in data.get("tools"):
                                tmp_function = {
                                    "type": "function",
                                    "function": {
                                        "name": server_name + "__" + item["name"],
                                        "description": item["description"],
                                        "parameters": {
                                            **item["parameters"],
                                            "properties": {
                                                k: v
                                                for k, v in item["parameters"]
                                                .get("properties", {})
                                                .items()
                                                if "default" not in v
                                            },
                                        },
                                    },
                                }
                                openai_tools.append(tmp_function)
                except Exception as e:
                    logger.warning(f"server_name:{server_name} translate failed: {e}")

            elif server_type == "sse":
                server_configs.append({
                    "name": server_name,
                    "type": "sse",
                    "params": {
                        "url": server_config["url"],
                        "headers": server_config.get("headers"),
                        "timeout": server_config.get("timeout"),
                        "sse_read_timeout": server_config.get("sse_read_timeout"),
                        "client_session_timeout_seconds": server_config.get("client_session_timeout_seconds")
                    },
                })

            elif server_type == "streamable-http":
                server_configs.append({
                    "name": server_name,
                    "type": "streamable-http",
                    "params": {
                        "url": server_config["url"],
                        "headers": server_config.get("headers"),
                        "timeout": server_config.get("timeout"),
                        "sse_read_timeout": server_config.get("sse_read_timeout"),
                        "client_session_timeout_seconds": server_config.get("client_session_timeout_seconds")
                    },
                })

            else:
                # Default to stdio
                server_configs.append({
                    "name": server_name,
                    "type": "stdio",
                    "params": {
                        "command": server_config["command"],
                        "args": server_config.get("args", []),
                        "env": server_config.get("env", {}),
                        "cwd": server_config.get("cwd"),
                        "encoding": server_config.get("encoding", "utf-8"),
                        "encoding_error_handler": server_config.get(
                            "encoding_error_handler", "strict"
                        ),
                        "client_session_timeout_seconds": server_config.get("client_session_timeout_seconds")
                    },
                })

    if not server_configs:
        return openai_tools

    # Load all MCP servers in parallel
    tasks = [
        _load_single_mcp_server(
            server_config=cfg,
            context=context,
            sandbox_id=sandbox_id,
            black_tool_actions=black_tool_actions,
            tool_actions=tool_actions
        )
        for cfg in server_configs
    ]

    logger.info(f"Loading {len(tasks)} MCP servers in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    mcp_openai_tools = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"MCP server '{server_configs[i]['name']}' failed: {result}")
        elif isinstance(result, list):
            mcp_openai_tools.extend(result)

    if mcp_openai_tools:
        openai_tools.extend(mcp_openai_tools)

    logger.info(f"Loaded {len(openai_tools)} tools from {len(server_configs)} MCP servers")
    return openai_tools


def enable_parallel_loading():
    """
    Monkey-patch utils.mcp_tool_desc_transform_v2 with the parallel version.

    Call this at application startup to enable parallel MCP loading.
    """
    global _original_mcp_tool_desc_transform_v2

    import aworld.mcp_client.utils as utils_module

    if _original_mcp_tool_desc_transform_v2 is None:
        _original_mcp_tool_desc_transform_v2 = utils_module.mcp_tool_desc_transform_v2

    utils_module.mcp_tool_desc_transform_v2 = mcp_tool_desc_transform_v2_parallel
    logger.info("Parallel MCP loading enabled")


def disable_parallel_loading():
    """
    Restore the original sequential mcp_tool_desc_transform_v2.
    """
    global _original_mcp_tool_desc_transform_v2

    if _original_mcp_tool_desc_transform_v2 is not None:
        import aworld.mcp_client.utils as utils_module
        utils_module.mcp_tool_desc_transform_v2 = _original_mcp_tool_desc_transform_v2
        logger.info("Parallel MCP loading disabled, restored original")


# Auto-enable if environment variable is set
if os.environ.get("AWORLD_PARALLEL_MCP", "").lower() in ("1", "true", "yes"):
    enable_parallel_loading()
