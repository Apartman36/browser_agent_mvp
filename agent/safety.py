from __future__ import annotations
import re

import json
from typing import Any
from agent.llm import LLMClient

def is_high_risk(action: dict[str, Any], current_obs: dict[str, Any]) -> tuple[bool, str]:
    if action.get("needs_user_confirmation") is True:
        return True, "planner requested user confirmation"

    tool = action.get("tool")
    if tool in {"observe", "query_page", "screenshot", "extract_text", "extract_dom", "extract_css", "ask_user", "done", "wait", "scroll", "goto"}:
        return False, ""


    try:
        llm = LLMClient()
        if not llm.client:
            return _regex_fallback(action, current_obs)

        messages = [
            {"role": "system", "content": "You are a safety classification sub-agent. Given the current observation and an action, determine if the action is high risk (modifies state irreversibly, submits forms, buys, deletes, etc). Return JSON only: {\"high_risk\": true/false, \"reason\": \"string\"}"},
            {"role": "user", "content": json.dumps({"action": action, "observation_snippet": current_obs.get("snapshot_yaml", "")[:5000]}, ensure_ascii=False)}
        ]

        # Use the verifier model for safety checks
        model = llm.verifier_model
        response = llm._chat_completion_with_retries(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        import agent.llm
        if hasattr(agent.llm, "strip_json_fences"):
            content = agent.llm.strip_json_fences(content)
        # Even without strip_json_fences, usually json mode guarantees json
        if "```" in content:
            # Fallback simple stripping
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]

        data = json.loads(content)
        if data.get("high_risk", False):
            return True, data.get("reason", "LLM determined high risk")
        return False, ""
    except Exception as e:
        print(f"Safety LLM check failed: {e}. Falling back to regex.")
        return _regex_fallback(action, current_obs)


DANGEROUS_RE = re.compile(
    r"(?i)(apply|pay|send|confirm|delete|remove|buy|checkout|submit application|отправ|отклик|подтверд|удал|оплат|купить|заказ)"
)

def _action_contains_dangerous_text(obj: Any) -> bool:
    if isinstance(obj, str):
        return bool(DANGEROUS_RE.search(obj))
    elif isinstance(obj, dict):
        return any(_action_contains_dangerous_text(k) or _action_contains_dangerous_text(v) for k, v in obj.items())
    elif isinstance(obj, list):
        return any(_action_contains_dangerous_text(item) for item in obj)
    return False

def _regex_fallback(action: dict[str, Any], current_obs: dict[str, Any]) -> tuple[bool, str]:
    tool = action.get("tool")
    args = action.get("args") or {}

    if tool == "click_element":
        ref = str(args.get("ref", ""))
        line = _snapshot_line_for_ref(current_obs.get("snapshot_yaml", ""), ref)
        if DANGEROUS_RE.search(line):
            return True, f'click on "{line.strip()}"'

    if tool == "type_text" and bool(args.get("submit", False)):
        ref = str(args.get("ref", ""))
        context = _snapshot_context_for_ref(current_obs.get("snapshot_yaml", ""), ref)
        if DANGEROUS_RE.search(context):
            return True, "typing text and pressing Enter may submit a high-risk form"

    if tool == "press_key" and str(args.get("key", "")).lower() == "enter":
        if _action_contains_dangerous_text(action):
            return True, "pressing Enter appears related to a high-risk action"

    return False, ""

def user_confirmed(answer: str) -> bool:
    return answer.strip().lower().startswith(("y", "yes", "д", "да"))

def _snapshot_line_for_ref(snapshot_yaml: str, ref: str) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"
    idx = snapshot_yaml.find(token)
    if idx == -1:
        return ""
    start = snapshot_yaml.rfind('\n', 0, idx)
    start = start + 1 if start != -1 else 0
    end = snapshot_yaml.find('\n', idx)
    end = end if end != -1 else len(snapshot_yaml)
    return snapshot_yaml[start:end]

def _snapshot_context_for_ref(snapshot_yaml: str, ref: str, radius: int = 2) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"
    idx = snapshot_yaml.find(token)
    if idx == -1:
        return ""
    start = idx
    for _ in range(radius + 1):
        start = snapshot_yaml.rfind('\n', 0, start)
        if start == -1:
            start = 0
            break
    if start != 0:
        start += 1
    end = idx
    for _ in range(radius + 1):
        next_end = snapshot_yaml.find('\n', end + 1) if end < len(snapshot_yaml) else -1
        if next_end == -1:
            end = len(snapshot_yaml)
            break
        end = next_end
    return snapshot_yaml[start:end]
