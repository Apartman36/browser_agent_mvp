import pytest

from agent.browser import Browser


class FakeLocator:
    def __init__(self, text: str) -> None:
        self.text = text

    def inner_text(self, timeout: int) -> str:
        return self.text


class FakePage:
    def __init__(self) -> None:
        self.selectors: list[str] = []

    def locator(self, selector: str) -> FakeLocator:
        self.selectors.append(selector)
        if selector == "body":
            return FakeLocator("full page body")
        return FakeLocator("specific ref")


@pytest.mark.parametrize("ref", [None, "None", "null", ""])
def test_extract_text_normalizes_empty_refs_to_full_page(ref: str | None) -> None:
    page = FakePage()
    browser = Browser()
    browser.page = page

    result = browser.extract_text(ref)

    assert result["ok"] is True
    assert result["data"]["ref"] is None
    assert result["data"]["text"] == "full page body"
    assert page.selectors == ["body"]
