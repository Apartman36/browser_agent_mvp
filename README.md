# Browser Automation AI Agent MVP

## What it is

A local autonomous browser agent in Python. The user gives a natural-language task, the agent opens visible Chromium, observes the current page through Playwright ARIA snapshots, asks an LLM for one structured next action, applies safety checks, executes a generic browser tool, and repeats until done.

This is a generic browser agent, not a site-specific bot.

## Demo

Demo video and repository links are provided in the submission message.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
```

## Browser/runtime requirements

- The primary runtime opens a visible Playwright Chromium window, not a headless-only browser.
- Browser state is persisted in `.pw_profile` through Playwright `launch_persistent_context`.
- A user can log in manually in that visible browser session, then let the agent continue in the same session.
- The agent observes pages with Playwright ARIA snapshots and acts only through current `aria-ref` values from the latest observation.
- Re-observe after navigation or mutation; refs are ephemeral and can become stale when the page changes.

## Browser runtime modes

The agent supports two launch modes. Defaults preserve the historical behavior, so doing nothing keeps the existing Playwright Chromium flow.

### Default: bundled Chromium

```env
BROWSER_MODE=chromium
```

- Playwright launches its bundled Chromium binary.
- Persistent profile directory: `.pw_profile` (or `CHROMIUM_USER_DATA_DIR` if set).
- No Chrome channel is passed to Playwright.
- This is the recommended mode for tests and for the most reproducible demo.

### Optional: installed Google Chrome with a separate agent profile

```env
BROWSER_MODE=chrome_profile
CHROME_CHANNEL=chrome
CHROME_USER_DATA_DIR=C:\Users\<you>\browser-agent-chrome-profile
```

- Playwright launches the locally installed Google Chrome via `channel="chrome"`.
- Persistent profile directory is a **separate** agent profile path you control.
- The agent never points Playwright at the user's main Chrome profile.
- First run: use `--login-wait` to log in manually inside the agent profile. Subsequent runs reuse cookies and `localStorage` from that profile.
- Do not put banking, government, primary email, or other sensitive accounts in the agent profile.
- This is purely about running inside real Chrome with a controlled profile. It is **not** stealth, anti-detection, or anti-bot evasion. Sites may still detect automation patterns or disallow automated actions.
- Respect site Terms of Service. Use the HITL confirmation flow for final actions such as Send/Post/Apply/Buy/Delete.
- If Chrome is not installed, the agent prints a clear error: `Chrome channel not available. Install Google Chrome or use BROWSER_MODE=chromium.`

### Optional action pacing and slow_mo

```env
BROWSER_SLOW_MO_MS=150
ACTION_MIN_DELAY_MS=800
ACTION_MAX_DELAY_MS=2500
```

- `BROWSER_SLOW_MO_MS` is Playwright's built-in `slow_mo`. Used for observability/stability, not stealth. Set to `0` to disable.
- `ACTION_MIN_DELAY_MS` and `ACTION_MAX_DELAY_MS` add an optional random pre-action sleep before `goto`, `click_element`, `type_text`, `press_key`, and `scroll`. Set both to `0` to disable.
- If both delays are `0`, the default flow runs with no extra sleeping.
- If `max < min`, the agent raises a clear `ValueError` at config load.

### PowerShell example — first login into the agent Chrome profile

```powershell
$env:BROWSER_MODE = "chrome_profile"
$env:CHROME_USER_DATA_DIR = "C:\Users\hustlePC\browser-agent-chrome-profile"
$env:BROWSER_SLOW_MO_MS = "150"
$env:ACTION_MIN_DELAY_MS = "800"
$env:ACTION_MAX_DELAY_MS = "2500"

.\.venv\Scripts\python.exe run.py --start-url https://example.com --max-steps 3 "Open example.com, read the page title, and finish."
```

Manual-login pattern (open a site, pause for you to sign in, then observe):

```powershell
.\.venv\Scripts\python.exe run.py --start-url https://example.com --login-wait --max-steps 3 "Open the site and wait for me to log in manually. Then observe the page and stop."
```

Generic browser assistant example (GitHub docs page):

```powershell
.\.venv\Scripts\python.exe run.py --start-url https://docs.github.com --max-steps 4 "Open the GitHub docs, read the top of the page, and finish with a one-line summary."
```

## Per-run logs

Each `run_agent` invocation creates its own directory under `RUN_LOG_ROOT` (default `logs/runs`).

```
logs/
  runs/
    2026-05-16_21-43-12/
      metadata.json
      actions.jsonl
      safety_audit.jsonl
      final_report.md
      screenshots/
        20260516_214315_123456.png
```

`metadata.json` includes the run id, started/ended timestamps, status, goal, planner mode, model, browser mode, browser channel (when applicable), profile dir, slow_mo, configured action delays, max steps, and start URL. No API keys, cookies, or `.env` contents are written there.

The CLI prints `Saved run logs: logs\runs\<run_id>` when the run finishes.

Copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`, then replace the model
placeholders with OpenRouter model IDs verified for your account.

Model configuration shape:

```env
PLANNER_MODE=auto
MODEL=<your-openrouter-model-id>
MODEL_FALLBACKS=<comma-separated-openrouter-model-ids>
PAID_FALLBACK_MODEL=<optional-paid-openrouter-model-id>
MODEL_VERIFIER=<optional-openrouter-model-id>
USE_LLM_RISK_CLASSIFIER=false
```

Free OpenRouter providers can rate-limit or fail upstream, so the client retries provider
errors and then tries `MODEL_FALLBACKS` in order. To force one model only, set `MODEL`
and leave `MODEL_FALLBACKS` empty. The repository tests do not validate specific
OpenRouter model IDs.

The LLM client is OpenAI-compatible. LM Studio/Ollama support could be added later by
making `base_url` configurable, but this MVP uses OpenRouter for a reproducible demo.

### Pre-flight on Windows (PowerShell)

Make sure the terminal and Python are both in UTF-8 mode so Russian text in
`logs/actions.jsonl` and `logs/final_report.md` is not corrupted into mojibake
(e.g. `РџРѕСЃ...`). Run once per shell before launching the agent:

```powershell
chcp 65001            # console code page to UTF-8
$env:PYTHONUTF8 = "1" # Python wide UTF-8 mode for this session
$env:PYTHONIOENCODING = "utf-8"
```

`run.py` also reconfigures `sys.stdin/stdout/stderr` to UTF-8 at startup, so
`input()` answers and prompts are safe even if you forget the variables above.

```powershell
python run.py --start-url https://hh.ru --login-wait "Найди 2 вакансии AI engineer в Москве на hh.ru, изучи описание и подготовь короткие сопроводительные письма. Перед отправкой откликов обязательно спроси подтверждение."
```

## Architecture in 60 seconds

`observe -> decide -> safety gate -> act -> memory/logs -> repeat`

1. Playwright observes the page with `page.locator("body").aria_snapshot(mode="ai")`.
2. The planner chooses exactly one next tool action, using native tool/function calling when `PLANNER_MODE=auto` or `native_tools`.
3. The `ToolRegistry` validates tool arguments from Pydantic schemas before dispatch.
4. `SafetyEngine` classifies the action by category and risk, then blocks or asks for confirmation when needed.
5. Browser tools execute through current ARIA refs such as `aria-ref=e7`.
6. A `query_page` sub-agent answers focused questions about the current page (titles, ref tables, salaries) without bloating the main loop.
7. Memory stores compact facts and only the last 8 action results.
8. Logs, safety audit records, and the final report are written under `logs/` as UTF-8.

## LLM interface

The primary planner path uses OpenAI-compatible Chat Completions tool/function
calling through OpenRouter. `NativeToolPlanner` sends `tools=ToolRegistry.openai_tools()`,
uses `tool_choice="auto"`, and requests one tool call per step with
`parallel_tool_calls=False` when supported.

When a native tool call is selected, the planner keeps the provider transcript
valid by storing the assistant `tool_calls` message and returning the execution
result as a `role: "tool"` message keyed by `tool_call_id`. Blocked and
user-denied actions are also returned to the provider as compact tool results.

`PLANNER_MODE=json` forces the compatibility planner. `PLANNER_MODE=auto` prefers
native tool calls and keeps JSON mode as a fallback for models/providers that do
not reliably support native tool calls. JSON mode remains a structured JSON
fallback and does not use provider `role: "tool"` messages. Both planner paths
use the same registry for tool descriptions and argument validation.

## Optional MCP server

This repository includes an optional MCP server that exposes a subset of the generic browser tools over the Model Context Protocol via stdio. The main CLI agent remains the primary demo runtime and is not itself an MCP client.

Current scope:

- server-side MCP support only;
- stdio transport only;
- generic browser tools only, with no site-specific workflows;
- one visible Playwright/Chromium session per MCP server process;
- one active page/tab at a time;
- browser startup is lazy on the first tool call;
- mutating MCP actions still pass through `SafetyEngine`;
- high-risk or confirmation-required MCP actions return a structured blocked result instead of executing silently.

Run the server:

```powershell
python -m agent.mcp_server
```

Test with MCP Inspector:

```powershell
npx -y @modelcontextprotocol/inspector python -m agent.mcp_server
```

Exposed MCP tools:

- `browser_observe`
- `browser_goto`
- `browser_click_element`
- `browser_type_text`
- `browser_extract_text`
- `browser_screenshot`
- `browser_scroll`
- `browser_wait`

Intentionally not exposed through MCP:

- `ask_user`
- `done`
- `query_page`
- arbitrary code execution
- site-specific flows

The MCP layer is a thin adapter over the existing browser/session/tool/safety code. It does not replace `NativeToolPlanner`, JSON fallback mode, the CLI HITL flow, or the Playwright ARIA-ref approach.

## Why this is not hardcoded

- No site-specific selectors.
- No task-specific workflows.
- No hardcoded hh.ru logic in `agent/`.
- No fixed paths, buttons, or page structures.
- The agent must choose refs from the current Playwright ARIA snapshot at runtime.
- Re-observation happens after each step because refs can become stale.

Grep proof:

```bash
grep -ri "hh.ru\|вакан\|data-qa\|querySelector\|css=" agent/
```

PowerShell equivalent:

```powershell
Select-String -Path .\agent\*.py -Pattern "hh.ru|вакан|data-qa|querySelector|css=|xpath" -CaseSensitive:$false
```

Expected result for `agent/`: empty.

Forbidden-framework check (must also be empty):

```powershell
Select-String -Path .\agent\*.py -Pattern "browser_use|skyvern|stagehand|selenium" -CaseSensitive:$false
```

## Tool surface

| Tool | Type | Args | Description |
| --- | --- | --- | --- |
| `goto` | Mutating | `{"url": str}` | Navigate current tab to a URL. |
| `observe` | Read-only | `{}` | Refresh the current page observation. |
| `query_page` | Read-only | `{"question": str}` | Ask a DOM/page analyst sub-agent about the current page. |
| `click_element` | Mutating | `{"ref": str}` | Click an element by current ARIA ref. |
| `type_text` | Mutating | `{"ref": str, "text": str, "submit": bool, "clear": bool}` | Fill or type text into an element, optionally pressing Enter. |
| `press_key` | Mutating | `{"key": str}` | Press a keyboard key. |
| `scroll` | Mutating | `{"direction": "up" \| "down"}` | Scroll the page. |
| `wait` | Read-only | `{"ms": int}` | Wait for a bounded duration. |
| `screenshot` | Read-only | `{"full_page": bool}` | Save a screenshot under `logs/screenshots/`. |
| `extract_text` | Read-only | `{"ref": str \| null}` | Extract visible text from a ref or full page. |
| `ask_user` | Read-only | `{"question": str}` | Ask the human for missing info or confirmation. |
| `done` | Read-only | `{"summary": str, "status": str}` | Finish and write final report. |

## Advanced patterns

- Native tool/function calling with a validated JSON-mode fallback.
- Tool registry as the single source of truth for schemas, prompt docs, categories, and risk metadata.
- Structured `SafetyEngine` with human confirmation for high-risk actions.
- Page analyst sub-agent through `query_page`.
- Compact memory and context management with bounded history.
- Optional MCP server support as a thin server-side adapter over the generic browser tool layer.

## Research / inspiration

Implemented here: direct Playwright automation plus a local Python tool dispatcher.
Optional MCP server support is included as a thin adapter over the existing
browser tool layer. The project does not depend on Playwright MCP.

- Playwright MCP inspired snapshot/ref-style interaction.
- browser-use inspired indexed and serialized page representation.
- Stagehand inspired the act/observe/extract mental model.
- Skyvern inspired safety notes and production workflow thinking.

No code was copied from those projects.

## Demo scenario

hh.ru flow:

1. User opens the browser and logs in manually.
2. Agent searches for AI engineer roles.
3. Agent opens 2 relevant results.
4. Agent extracts title, company, salary if visible, and top requirements.
5. Agent prepares short Russian cover letters.
6. Agent stops before clicking Apply/Oткликнуться unless the user explicitly confirms.

## Safety

- Every planned tool call passes through `ToolRegistry` validation and `SafetyEngine`.
- Actions are classified by category and risk before dispatch.
- Regex/keyword checks are heuristic signals inside the policy, not the whole safety layer.
- High-risk submit/send/apply/delete/pay/buy actions require human confirmation.
- Prompt-injection-looking page text escalates mutating actions.
- Safety decisions are written to `logs/safety_audit.jsonl` with sensitive typed values redacted.
- No CAPTCHA solving, password extraction, or automatic irreversible actions.

## Known limitations

- Single tab.
- No vision fallback.
- No CAPTCHA solving.
- ARIA snapshot quality depends on website accessibility.
- MCP server support is minimal and stdio-only.
- The MCP server has one active session/page per process and does not implement multi-session routing.
- The CLI agent is not an MCP client.
- MCP resources/prompts and HTTP transport are not implemented.
- High-risk MCP actions return confirmation-required results instead of using the CLI approval prompt.
- Free OpenRouter models may rate-limit or produce weaker native/JSON actions; retries and fallback
  models are implemented for demo robustness.
- LM Studio/Ollama local provider wiring is future work.

## Useful commands

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Optional OpenRouter smoke test:

```powershell
python scripts/smoke_openrouter.py
```

Dry run without an LLM call:

```powershell
python run.py --start-url https://www.google.com --dry-run "Observe the page."
```

Run the optional MCP server:

```powershell
python -m agent.mcp_server
```

Inspect the MCP server over stdio:

```powershell
npx -y @modelcontextprotocol/inspector python -m agent.mcp_server
```
