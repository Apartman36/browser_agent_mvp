from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from agent.memory import Memory
from agent.planners.base import PlannerAction
from agent.tool_registry import ActionCategory, RiskLevel, ToolSpec


DANGEROUS_LABEL_RE = re.compile(
    r"(?i)(apply|pay|send|confirm|delete|remove|buy|checkout|submit application|place order|"
    r"publish|post|authorize|grant access|"
    r"отправ|отклик|подтверд|удал|оплат|купить|заказ|сохранить|изменить пароль|"
    r"РѕС‚РїСЂР°РІ|РѕС‚РєР»РёРє|РїРѕРґС‚РІРµСЂРґ|СѓРґР°Р»|РѕРїР»Р°С‚|"
    r"РєСѓРїРёС‚СЊ|Р·Р°РєР°Р·|РћС‚РєР»РёРє)"
)
BENIGN_SEARCH_RE = re.compile(
    r"(?i)(search textbox|search box|search field|search jobs|search|find|filter|apply filter|"
    r"найти|поиск|фильтр|применить фильтр|"
    r"РќР°Р№С‚Рё|РџРѕРёСЃРє|РџСЂРёРјРµРЅРёС‚СЊ С„РёР»СЊС‚СЂ|"
    r"Профессия, должность или компания|РџСЂРѕС„РµСЃСЃРёСЏ)"
)
PROMPT_INJECTION_RE = re.compile(
    r"(?i)(ignore previous instructions|system prompt|reveal secrets|click apply automatically|"
    r"developer message|tool instructions|exfiltrate|steal)"
)
SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(password|passwd|token|api[_-]?key|secret|bearer\s+[a-z0-9._-]+|"
    r"sk-[a-z0-9_-]{12,}|[0-9]{13,19})"
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    requires_confirmation: bool
    blocked: bool
    risk: RiskLevel
    category: ActionCategory
    reason: str
    policy_rule: str


class SafetyEngine:
    def evaluate(
        self,
        action: PlannerAction,
        tool_spec: ToolSpec,
        observation: dict[str, Any],
        memory: Memory | None = None,
    ) -> SafetyDecision:
        if self._is_repeated_action(action, memory):
            return self._blocked(
                tool_spec,
                RiskLevel.HIGH,
                "same tool and arguments repeated too often",
                "loop_detection",
            )

        if tool_spec.read_only:
            return self._allowed(tool_spec, RiskLevel.LOW, "read-only tool", "read_only_default")

        if action.needs_user_confirmation:
            return self._confirm(
                tool_spec,
                RiskLevel.HIGH,
                "planner requested user confirmation",
                "planner_confirmation",
            )

        if action.tool == "goto":
            return self._evaluate_goto(action, tool_spec)

        if action.tool == "scroll":
            return self._allowed(tool_spec, RiskLevel.LOW, "navigation scroll", "navigation_default")

        suspicious_page = self._has_prompt_injection(observation)
        if action.tool == "type_text":
            text = str(action.args.get("text", ""))
            if self._looks_sensitive(text):
                return self._confirm(
                    tool_spec,
                    RiskLevel.HIGH,
                    "typed text looks sensitive",
                    "sensitive_text",
                )

        if tool_spec.category == ActionCategory.REVERSIBLE:
            ref_context = self._ref_context(observation, str(action.args.get("ref", "")))
            action_text = json.dumps(action.args, ensure_ascii=False, default=str)
            combined = f"{ref_context}\n{action_text}"
            if suspicious_page:
                return self._confirm(
                    tool_spec,
                    RiskLevel.HIGH,
                    "page content contains prompt-injection signals",
                    "prompt_injection_escalation",
                )
            if self._is_dangerous_context(combined) and not self._is_benign_search_context(combined):
                risk = RiskLevel.CRITICAL if re.search(r"(?i)(delete|remove|pay|buy|checkout|place order|удал|оплат|купить)", combined) else RiskLevel.HIGH
                return self._confirm(
                    tool_spec,
                    risk,
                    f"target context suggests high-impact action: {ref_context.strip()}",
                    "dangerous_action_label",
                )
            return self._allowed(tool_spec, RiskLevel.MEDIUM, "reversible browser interaction", "reversible_default")

        if tool_spec.category in {
            ActionCategory.MUTATING,
            ActionCategory.IRREVERSIBLE,
            ActionCategory.SENSITIVE,
        }:
            return self._confirm(tool_spec, RiskLevel.HIGH, "state-changing action", "mutating_default")

        return self._allowed(tool_spec, tool_spec.default_risk, "default allow", "default")

    def _evaluate_goto(self, action: PlannerAction, tool_spec: ToolSpec) -> SafetyDecision:
        raw_url = str(action.args.get("url", "")).strip()
        parsed = urlparse(raw_url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return self._blocked(
                tool_spec,
                RiskLevel.CRITICAL,
                f"blocked non-http URL scheme: {parsed.scheme or 'missing'}",
                "navigation_scheme_block",
            )
        return self._allowed(tool_spec, RiskLevel.MEDIUM, "http navigation", "navigation_http")

    @staticmethod
    def _has_prompt_injection(observation: dict[str, Any]) -> bool:
        text = "\n".join(
            [
                str(observation.get("snapshot_yaml", "")),
                str(observation.get("body_text", "")),
                str(observation.get("title", "")),
            ]
        )
        return bool(PROMPT_INJECTION_RE.search(text))

    @staticmethod
    def _looks_sensitive(text: str) -> bool:
        return bool(SENSITIVE_TEXT_RE.search(text))

    @staticmethod
    def _is_dangerous_context(text: str) -> bool:
        return bool(DANGEROUS_LABEL_RE.search(text))

    @staticmethod
    def _is_benign_search_context(text: str) -> bool:
        return bool(BENIGN_SEARCH_RE.search(text))

    @staticmethod
    def _ref_context(observation: dict[str, Any], ref: str, radius: int = 2) -> str:
        snapshot = str(observation.get("snapshot_yaml", ""))
        if not ref:
            return ""
        token = f"[ref={ref}]"
        lines = snapshot.splitlines()
        for index, line in enumerate(lines):
            if token in line:
                start = max(0, index - radius)
                end = min(len(lines), index + radius + 1)
                return "\n".join(lines[start:end])
        return ""

    @staticmethod
    def _is_repeated_action(action: PlannerAction, memory: Memory | None) -> bool:
        if memory is None:
            return False
        recent = list(memory.history)[-5:]
        current_key = (action.tool, _stable_args(action.args))
        matches = 0
        for item in recent:
            if (item.get("tool"), _stable_args(item.get("args", {}))) == current_key:
                matches += 1
        return matches >= 2

    @staticmethod
    def _allowed(tool_spec: ToolSpec, risk: RiskLevel, reason: str, policy_rule: str) -> SafetyDecision:
        return SafetyDecision(
            allowed=True,
            requires_confirmation=False,
            blocked=False,
            risk=risk,
            category=tool_spec.category,
            reason=reason,
            policy_rule=policy_rule,
        )

    @staticmethod
    def _confirm(tool_spec: ToolSpec, risk: RiskLevel, reason: str, policy_rule: str) -> SafetyDecision:
        return SafetyDecision(
            allowed=True,
            requires_confirmation=True,
            blocked=False,
            risk=risk,
            category=tool_spec.category,
            reason=reason,
            policy_rule=policy_rule,
        )

    @staticmethod
    def _blocked(tool_spec: ToolSpec, risk: RiskLevel, reason: str, policy_rule: str) -> SafetyDecision:
        return SafetyDecision(
            allowed=False,
            requires_confirmation=False,
            blocked=True,
            risk=risk,
            category=tool_spec.category,
            reason=reason,
            policy_rule=policy_rule,
        )


def _stable_args(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args)
    normalized = {str(key): str(value) for key, value in sorted(args.items(), key=lambda item: str(item[0]))}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
