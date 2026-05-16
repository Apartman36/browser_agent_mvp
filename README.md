# Browser Automation AI Agent MVP

## What it is

A local autonomous browser agent in Python. The user gives a natural-language task, the agent opens visible Chromium, observes the current page through Playwright ARIA snapshots, asks an LLM for one structured next action, applies safety checks, executes a generic browser tool, and repeats until done.

This is a generic browser agent, not a site-specific bot.

## Demo

Video: TODO
GitHub: TODO

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY`.

Recommended model settings:

```env
MODEL=google/gemma-4-31b-it:free
MODEL_FALLBACKS=google/gemma-4-26b-a4b-it:free,z-ai/glm-4.5-air:free,inclusionai/ring-2.6-1t:free
MODEL_VERIFIER=google/gemma-4-31b-it:free
```

Free OpenRouter providers can rate-limit or fail upstream, so the client retries provider
errors and then tries `MODEL_FALLBACKS` in order. To force one model only, set `MODEL`
and leave `MODEL_FALLBACKS` empty.

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
2. The planner LLM returns exactly one JSON action.
3. The safety gate blocks irreversible actions until the user confirms.
4. Browser tools execute through current ARIA refs such as `aria-ref=e7`.
5. A `query_page` sub-agent answers focused questions about the current page (titles, ref tables, salaries) without bloating the main loop.
6. Memory stores compact facts and only the last 8 action results.
7. Logs and final report are written under `logs/` as UTF-8 (JSONL with `ensure_ascii=False`).

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

- Structured JSON action schema with validation and retry on invalid JSON.
- Human-in-the-loop safety gate for submit/send/apply/delete/pay/buy actions.
- Page analyst sub-agent through `query_page`.
- Compact memory and context management with bounded history.

## Research / inspiration

- Playwright uses ARIA snapshot/ref-style interaction (Note: This is not related to the Model Context Protocol / MCP).
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

- No CAPTCHA solving.
- No password extraction.
- No automatic submit/payment/delete/apply without confirmation.
- No API keys, cookies, tokens, or secrets are printed.

## Known limitations

- Single tab.
- No vision fallback.
- No CAPTCHA solving.
- ARIA snapshot quality depends on website accessibility.
- Free OpenRouter models may rate-limit or produce weaker JSON; retries and fallback
  models are implemented for demo robustness.
- LM Studio/Ollama local provider wiring is future work.

## Useful commands

Run tests:

```powershell
pytest -q
```

Optional OpenRouter smoke test:

```powershell
python scripts/smoke_openrouter.py
```

Dry run without an LLM call:

```powershell
python run.py --start-url https://www.google.com --dry-run "Observe the page."
```
