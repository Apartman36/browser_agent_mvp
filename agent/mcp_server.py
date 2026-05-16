"""
Basic MCP (Model Context Protocol) Server implementation exposing browser automation tools.
"""

from typing import Any
import json
import sys
from agent.tools import TOOL_DESCRIPTIONS, ToolDispatcher

class MCPServer:
    def __init__(self, browser: Any, llm_client: Any):
        self.tools = ToolDispatcher(browser, llm_client)

    def handle_request(self, request_str: str) -> str:
        try:
            req = json.loads(request_str)
            method = req.get("method")
            if method == "list_tools":
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": name,
                                "description": meta["description"],
                                "parameters": {
                                    "type": "object",
                                    "properties": {"args": {"type": "string", "description": "JSON serialized arguments"}}
                                }
                            }
                            for name, meta in TOOL_DESCRIPTIONS.items()
                        ]
                    }
                })
            elif method == "call_tool":
                params = req.get("params", {})
                name = params.get("name")
                args_str = params.get("arguments", {}).get("args", "{}")
                args = json.loads(args_str)
                action = {"tool": name, "args": args}
                result = self.tools.dispatch(action)
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req.get("id"),
                    "result": result
                })
            else:
                return json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32601, "message": "Method not found"}})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}})

    def run_stdio(self):
        print("MCP Server started. Awaiting JSON-RPC requests on stdin.", file=sys.stderr)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_request(line)
            print(response)
            sys.stdout.flush()
