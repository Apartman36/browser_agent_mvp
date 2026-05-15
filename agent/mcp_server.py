from __future__ import annotations

import atexit
import sys
from typing import Any, Literal

from agent.mcp_tools import MCP_TOOL_NAMES, MCPBrowserTools


def _load_fastmcp_class() -> type[Any]:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The optional MCP server requires the official Python MCP SDK. "
            "Install project requirements first: pip install -r requirements.txt"
        ) from exc
    return FastMCP


def create_mcp_server(tools: MCPBrowserTools | None = None) -> Any:
    """Create a FastMCP server without starting the browser or transport."""

    FastMCP = _load_fastmcp_class()
    browser_tools = tools or MCPBrowserTools()
    server = FastMCP("browser-agent-mvp")

    @server.tool()
    def browser_observe() -> dict[str, Any]:
        """Observe the current page through the Playwright ARIA snapshot."""

        return browser_tools.browser_observe()

    @server.tool()
    def browser_goto(url: str) -> dict[str, Any]:
        """Navigate the active page to an HTTP or HTTPS URL."""

        return browser_tools.browser_goto(url)

    @server.tool()
    def browser_click_element(ref: str) -> dict[str, Any]:
        """Click one element by ARIA ref from the latest browser_observe result."""

        return browser_tools.browser_click_element(ref)

    @server.tool()
    def browser_type_text(
        ref: str,
        text: str,
        submit: bool = False,
        clear: bool = True,
    ) -> dict[str, Any]:
        """Type text into one ARIA-ref element, optionally submitting with Enter."""

        return browser_tools.browser_type_text(ref=ref, text=text, submit=submit, clear=clear)

    @server.tool()
    def browser_extract_text(ref: str | None = None) -> dict[str, Any]:
        """Extract visible text from an ARIA ref, or from the full page when ref is null."""

        return browser_tools.browser_extract_text(ref)

    @server.tool()
    def browser_screenshot(full_page: bool = False) -> dict[str, Any]:
        """Save a screenshot under logs/screenshots and return the artifact path."""

        return browser_tools.browser_screenshot(full_page)

    @server.tool()
    def browser_scroll(direction: Literal["up", "down"]) -> dict[str, Any]:
        """Scroll the active page up or down."""

        return browser_tools.browser_scroll(direction)

    @server.tool()
    def browser_wait(ms: int) -> dict[str, Any]:
        """Wait for a bounded number of milliseconds."""

        return browser_tools.browser_wait(ms)

    setattr(server, "_browser_agent_tools", browser_tools)
    setattr(server, "_browser_agent_tool_names", MCP_TOOL_NAMES)
    return server


def main() -> None:
    tools = MCPBrowserTools()
    atexit.register(tools.close)
    try:
        server = create_mcp_server(tools)
        server.run(transport="stdio")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        tools.close()


if __name__ == "__main__":
    main()
