# hh.ru Demo Notes

## Recommended command

```powershell
python run.py --start-url https://hh.ru --login-wait "Найди 2 вакансии AI engineer в Москве на hh.ru, изучи описание, выпиши название, компанию, зарплату если есть и 3 главных требования. Для каждой подготовь короткое сопроводительное письмо на русском. Перед отправкой отклика обязательно спроси подтверждение."
```

## Before recording

- Start with a short dry run to confirm Chromium, refs, and UTF-8 output are healthy.
- Keep 2-3 fallback models in `.env` via `MODEL_FALLBACKS`.
- If a free model returns 429/502, rerun or switch `MODEL` in `.env`.

## Expected video flow

1. Start terminal on one side and visible Chromium on the other.
2. Run the command.
3. Browser opens `https://hh.ru`.
4. If needed, log in manually.
5. Press Enter in the terminal.
6. Show the agent observing the page, using refs from the ARIA snapshot, and logging each tool call.
7. Show `DOM Sub-agent: Processing query...` when the agent asks page-analysis questions.
8. Show the safety gate before any apply/send action.
9. Show the final structured report in the terminal and `logs/final_report.md`.

## OBS layout

- Left side: visible Chromium browser.
- Right side: PowerShell terminal with Rich logs.
- Optional bottom strip: project folder with `logs/actions.jsonl` and `logs/final_report.md`.

## Manual login

Use `--login-wait`. The browser opens first, then the terminal waits:

```text
Log in manually if needed, then press Enter
>
```

Complete login yourself. The agent must not ask for passwords, print secrets, solve CAPTCHA, or bypass account security.

## Final report should include

- Which results were opened.
- Title, company, salary if visible, and 3 main requirements for each result.
- Draft Russian cover letter for each result.
- Whether any apply/send action was requested.
- Confirmation that no application was submitted without explicit user approval.
