from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from agent.mcp_tools import FORBIDDEN_MCP_TOOL_NAMES, MCP_TOOL_NAMES, MCPBrowserTools


class FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.registered_tools: list[Callable[..., Any]] = []

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.registered_tools.append(func)
            return func

        return decorator


def test_mcp_server_module_imports_without_starting_browser() -> None:
    module = importlib.import_module("agent.mcp_server")
    module = importlib.reload(module)

    assert hasattr(module, "create_mcp_server")
    assert hasattr(module, "main")


def test_mcp_server_registers_only_allowed_browser_tools(monkeypatch) -> None:
    from agent import mcp_server

    monkeypatch.setattr(mcp_server, "_load_fastmcp_class", lambda: FakeFastMCP)
    server = mcp_server.create_mcp_server(MCPBrowserTools(browser_factory=lambda: object(), audit_writer=None))
    registered_names = {tool.__name__ for tool in server.registered_tools}

    assert registered_names == set(MCP_TOOL_NAMES)
    assert not registered_names.intersection(FORBIDDEN_MCP_TOOL_NAMES)


def test_requirements_include_official_mcp_sdk() -> None:
    text = Path("requirements.txt").read_text(encoding="utf-8")

    assert "mcp[cli]" in text


def test_readme_documents_optional_server_side_mcp_scope() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "optional MCP server" in text
    assert "server-side MCP support only" in text
    assert "not itself an MCP client" in text
    assert "python -m agent.mcp_server" in text
    assert "npx -y @modelcontextprotocol/inspector python -m agent.mcp_server" in text
    for name in MCP_TOOL_NAMES:
        assert name in text
    for name in FORBIDDEN_MCP_TOOL_NAMES:
        assert f"`{name}`" in text
