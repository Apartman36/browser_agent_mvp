import re

from playwright.sync_api import sync_playwright


def test_playwright_aria_refs_can_drive_clicks() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(
            """
            <html>
              <body>
                <button onclick="document.getElementById('status').textContent='clicked'">Run action</button>
                <div id="status" aria-live="polite">idle</div>
              </body>
            </html>
            """
        )
        snapshot = page.locator("body").aria_snapshot(mode="ai")
        assert "[ref=" in snapshot

        button_ref = None
        for line in snapshot.splitlines():
            if "Run action" in line:
                match = re.search(r"\[ref=([^\]]+)\]", line)
                if match:
                    button_ref = match.group(1)
                    break

        assert button_ref is not None
        page.locator(f"aria-ref={button_ref}").click(timeout=5000)
        assert page.locator("#status").inner_text() == "clicked"
        browser.close()

