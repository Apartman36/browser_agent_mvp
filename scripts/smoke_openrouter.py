from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY is missing. Create .env from .env.example and add the key.")
        return 1

    model = os.getenv("MODEL", "openai/gpt-oss-120b:free")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "browser-agent-mvp",
        },
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Return exactly valid JSON: {"ok": true, "message": "ready"}',
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

