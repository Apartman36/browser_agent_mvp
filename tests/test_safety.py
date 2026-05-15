from agent.safety import is_high_risk


def test_search_textbox_submit_is_not_high_risk() -> None:
    obs = {"snapshot_yaml": '- textbox "Профессия, должность или компания" [ref=e1]'}
    action = {
        "tool": "type_text",
        "args": {"ref": "e1", "text": "AI engineer", "submit": True, "clear": True},
        "needs_user_confirmation": False,
    }
    risky, reason = is_high_risk(action, obs)
    assert risky is False
    assert reason == ""


def test_find_button_click_is_not_high_risk() -> None:
    obs = {"snapshot_yaml": '- button "Найти" [ref=e2]'}
    action = {"tool": "click_element", "args": {"ref": "e2"}, "needs_user_confirmation": False}
    risky, reason = is_high_risk(action, obs)
    assert risky is False
    assert reason == ""


def test_dangerous_click_labels_are_high_risk() -> None:
    labels = ["Откликнуться", "Apply", "Delete", "Pay"]
    for label in labels:
        obs = {"snapshot_yaml": f'- button "{label}" [ref=e3]'}
        action = {"tool": "click_element", "args": {"ref": "e3"}, "needs_user_confirmation": False}
        risky, reason = is_high_risk(action, obs)
        assert risky, label
        assert reason


def test_submit_application_click_is_high_risk() -> None:
    obs = {"snapshot_yaml": '- button "Submit application" [ref=e4]'}
    action = {"tool": "click_element", "args": {"ref": "e4"}, "needs_user_confirmation": False}
    risky, reason = is_high_risk(action, obs)
    assert risky is True
    assert "Submit application" in reason


def test_planner_confirmation_flag_is_high_risk() -> None:
    obs = {"snapshot_yaml": ""}
    action = {"tool": "wait", "args": {"ms": 100}, "needs_user_confirmation": True}
    risky, reason = is_high_risk(action, obs)
    assert risky is True
    assert "confirmation" in reason


def test_press_enter_with_dangerous_context_is_high_risk() -> None:
    obs = {"snapshot_yaml": ""}
    action = {
        "tool": "press_key",
        "args": {"key": "enter", "comment": "confirming payment"},
        "needs_user_confirmation": False,
    }
    risky, reason = is_high_risk(action, obs)
    assert risky is True
    assert "pressing Enter" in reason
