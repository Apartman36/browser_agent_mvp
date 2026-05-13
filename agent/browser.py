from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class Browser:
    """Small Playwright wrapper that acts only through current ARIA refs."""

    def __init__(self, user_data_dir: str | Path = "./.pw_profile") -> None:
        self.user_data_dir = Path(user_data_dir)
        self.playwright = None
        self.context = None
        self.page = None

    def __enter__(self) -> "Browser":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1920, "height": 1080},
            user_agent=DEFAULT_USER_AGENT,
        )
        self.context.set_default_timeout(8000)
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.page.set_default_timeout(8000)

    def close(self) -> None:
        try:
            if self.context:
                self.context.close()
        finally:
            if self.playwright:
                self.playwright.stop()

    def observe(self) -> dict[str, Any]:
        page = self._page()
        snapshot = ""
        body_text = ""
        error = None

        try:
            snapshot = page.locator("body").aria_snapshot(mode="ai")
        except Exception as exc:  # Playwright can raise browser-specific errors.
            error = f"ARIA snapshot failed: {exc}"

        try:
            body_text = page.locator("body").inner_text(timeout=3000)
        except Exception as exc:
            if error:
                error = f"{error}; body text failed: {exc}"
            else:
                error = f"Body text failed: {exc}"

        return {
            "ok": error is None,
            "url": page.url,
            "title": self._safe_title(),
            "snapshot_yaml": self._truncate(snapshot, 20000),
            "body_text": self._truncate(body_text, 8000),
            "error": error,
        }

    def goto(self, url: str) -> dict[str, Any]:
        try:
            self._page().goto(url, wait_until="domcontentloaded", timeout=15000)
            self._settle()
            return self._ok("successfully navigated", {"url": self._page().url})
        except Exception as exc:
            return self._err("navigation failed", exc)

    def click_element(self, ref: str) -> dict[str, Any]:
        try:
            self._page().locator(f"aria-ref={ref}").click(timeout=8000)
            self._settle()
            return self._ok("clicked element", {"ref": ref, "url": self._page().url})
        except Exception as exc:
            return self._err(f"click failed for ref={ref}", exc)

    def type_text(
        self,
        ref: str,
        text: str,
        submit: bool = False,
        clear: bool = True,
    ) -> dict[str, Any]:
        try:
            locator = self._page().locator(f"aria-ref={ref}")
            if clear:
                locator.fill(text, timeout=8000)
            else:
                locator.click(timeout=8000)
                locator.press("End", timeout=8000)
                locator.type(text, timeout=8000)
            if submit:
                locator.press("Enter", timeout=8000)
            self._settle()
            return self._ok(
                "typed text",
                {"ref": ref, "submitted": submit, "chars": len(text), "url": self._page().url},
            )
        except Exception as exc:
            return self._err(f"type_text failed for ref={ref}", exc)

    def press_key(self, key: str) -> dict[str, Any]:
        try:
            self._page().keyboard.press(key)
            self._settle()
            return self._ok("pressed key", {"key": key, "url": self._page().url})
        except Exception as exc:
            return self._err(f"press_key failed for key={key}", exc)

    def scroll(self, direction: str) -> dict[str, Any]:
        try:
            if direction not in {"up", "down"}:
                return {
                    "ok": False,
                    "message": "invalid scroll direction",
                    "data": {"direction": direction},
                }
            delta = -900 if direction == "up" else 900
            self._page().mouse.wheel(0, delta)
            self._page().wait_for_timeout(500)
            return self._ok("scrolled page", {"direction": direction})
        except Exception as exc:
            return self._err("scroll failed", exc)

    def wait(self, ms: int) -> dict[str, Any]:
        try:
            bounded_ms = max(0, min(int(ms), 30000))
            self._page().wait_for_timeout(bounded_ms)
            self._settle(network_idle_ms=1000)
            return self._ok("waited", {"ms": bounded_ms})
        except Exception as exc:
            return self._err("wait failed", exc)

    def screenshot(self, full_page: bool = False) -> dict[str, Any]:
        try:
            screenshot_dir = Path("logs") / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            filename = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f.png")
            path = screenshot_dir / filename
            self._page().screenshot(path=str(path), full_page=full_page, timeout=10000)
            return self._ok("saved screenshot", {"path": str(path), "full_page": full_page})
        except Exception as exc:
            return self._err("screenshot failed", exc)

    def extract_text(self, ref: str | None = None) -> dict[str, Any]:
        try:
            locator = self._page().locator(f"aria-ref={ref}") if ref else self._page().locator("body")
            text = locator.inner_text(timeout=5000)
            return self._ok(
                "extracted text",
                {"ref": ref, "text": self._truncate(text, 12000), "chars": len(text)},
            )
        except Exception as exc:
            return self._err("extract_text failed", exc)

    def _page(self):
        if self.page is None:
            raise RuntimeError("Browser is not started.")
        return self.page

    def _safe_title(self) -> str:
        try:
            return self._page().title()
        except PlaywrightError:
            return ""

    def _settle(self, network_idle_ms: int = 2500) -> None:
        page = self._page()
        try:
            page.wait_for_load_state("domcontentloaded", timeout=3000)
        except PlaywrightError:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=network_idle_ms)
        except PlaywrightError:
            pass
        page.wait_for_timeout(300)

    @staticmethod
    def _ok(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ok": True, "message": message, "data": data or {}}

    @staticmethod
    def _err(message: str, exc: Exception) -> dict[str, Any]:
        return {"ok": False, "message": message, "data": {"error": str(exc)}}

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "\n...[truncated]"

