from agent.memory import Memory


def test_memory_keeps_bounded_history() -> None:
    memory = Memory("test goal")
    for idx in range(20):
        memory.add_action(
            {"thought": f"step {idx}", "tool": "wait", "args": {"ms": idx}},
            {"ok": True, "message": "ok", "data": {}},
        )
    payload = memory.to_prompt_payload()
    assert len(memory.history) == 8
    assert len(payload["recent_history"]) == 8
    assert payload["recent_history"][0]["step"] == 13


def test_facts_merge_and_values_are_strings() -> None:
    memory = Memory("test goal")
    memory.merge_facts({"answer": 42, "name": "Ada"})
    memory.merge_facts({"answer": "updated"})
    assert memory.facts == {"answer": "updated", "name": "Ada"}


def test_prompt_payload_truncates_large_observation() -> None:
    memory = Memory("test goal")
    memory.update_observation(
        {
            "url": "https://example.test",
            "title": "Example",
            "snapshot_yaml": "x" * 20000,
            "body_text": "y" * 10000,
        }
    )
    payload = memory.to_prompt_payload()
    assert payload["current_page"]["url"] == "https://example.test"
    assert len(payload["current_page"]["snapshot_yaml"]) < 12200
    assert len(payload["current_page"]["body_text_excerpt"]) < 6200

