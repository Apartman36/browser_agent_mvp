from agent.safety import is_high_risk


def test_dangerous_click_labels_are_high_risk() -> None:
    labels = ["Откликнуться", "Удалить", "Pay", "Submit"]
    for label in labels:
        obs = {"snapshot_yaml": f'- button "{label}" [ref=e1]'}
        action = {"tool": "click_element", "args": {"ref": "e1"}, "needs_user_confirmation": False}
        risky, reason = is_high_risk(action, obs)
        assert risky, label
        assert reason


def test_harmless_click_is_not_high_risk() -> None:
    obs = {"snapshot_yaml": '- button "Подробнее" [ref=e2]'}
    action = {"tool": "click_element", "args": {"ref": "e2"}, "needs_user_confirmation": False}
    risky, reason = is_high_risk(action, obs)
    assert risky is False
    assert reason == ""


def test_planner_confirmation_flag_is_high_risk() -> None:
    obs = {"snapshot_yaml": ""}
    action = {"tool": "wait", "args": {"ms": 100}, "needs_user_confirmation": True}
    risky, reason = is_high_risk(action, obs)
    assert risky is True
    assert "confirmation" in reason

