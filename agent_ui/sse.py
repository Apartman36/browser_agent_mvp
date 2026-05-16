from __future__ import annotations

import json
from typing import Any


def format_sse(event_type: str, data: dict[str, Any]) -> str:
    safe_event_type = event_type.replace("\r", "").replace("\n", "")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {safe_event_type}\ndata: {encoded}\n\n"
