# Changes

## Experimental local Web UI

This branch adds a small local dashboard for running the existing browser agent.
The CLI remains the primary stable entry point, and the optional MCP server
continues to run separately.

Run:

```powershell
python -m agent_ui --port 8765
```

Open the printed local URL:

```text
http://127.0.0.1:8765/?token=<printed-token>
```

The server binds to `127.0.0.1` only. The token is generated per launch unless
`AGENT_UI_TOKEN` is set. Do not expose this service on a public network.

Screenshots are never uploaded by the UI. If the existing screenshot tool saves
a file under `logs/screenshots`, the dashboard can show that local file through
a path-restricted localhost route.

Known limitations:

- One active run at a time.
- In-memory state only; no task queue, persistent database, or history.
- No multi-tab UI.
- No vision fallback.
- Safety confirmations can be answered in the dashboard for web-launched runs,
  while the CLI keeps its existing terminal prompt behavior.
- The `ask_user` tool still uses the existing terminal input path.
