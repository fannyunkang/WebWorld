"""Real browser demo for the observe -> decide -> execute loop.

This demo uses a local HTML page so it can run without network access or a
large model. The policy is mocked, but the browser observation and action
execution are real Playwright calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright

from mm_webagent.agent.action_parser import extract_action
from mm_webagent.browser.executor import PlaywrightActionExecutor
from mm_webagent.browser.observer import PlaywrightObserver


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Mini Shop</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 48px; max-width: 760px; }
    label, input, button { font-size: 18px; }
    input { padding: 10px; width: 320px; }
    button { padding: 10px 16px; margin-left: 8px; }
    #result { margin-top: 24px; font-size: 20px; font-weight: 700; }
  </style>
</head>
<body>
  <h1>Mini Shop</h1>
  <p>Search the store catalog.</p>
  <label for="query">Product</label>
  <input id="query" placeholder="type a product" />
  <button id="search" onclick="runSearch()">Search</button>
  <div id="result" role="status"></div>
  <script>
    function runSearch() {
      const value = document.querySelector('#query').value || 'nothing';
      document.querySelector('#result').textContent = 'Showing results for ' + value;
    }
  </script>
</body>
</html>
"""


def mock_policy(page_state: str) -> str:
    """Pretend to be a model choosing one browser action from the page state."""
    if "Showing results for backpack" in page_state:
        return "<reason>The requested result is visible.</reason><action>send_msg_to_user('Done')</action>"
    if "input 'backpack'" in page_state:
        return "<reason>The product is typed, so submit the form.</reason><action>click('2')</action>"
    if "input" in page_state and "button 'Search'" in page_state:
        return "<reason>Type the desired product first.</reason><action>fill('1', 'backpack')</action>"
    return "<reason>Submit the search form.</reason><action>click('2')</action>"


def main() -> None:
    output_dir = PROJECT_ROOT / "runs" / "real_browser_demo"
    observer = PlaywrightObserver(output_dir=output_dir)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 960, "height": 640})
        page.set_content(HTML)

        executor = PlaywrightActionExecutor(page=page)

        for step in range(4):
            observation = observer.observe(page, step_index=step)
            executor.element_registry = observation.element_registry

            print(f"\n--- Step {step} ---")
            print(observation.page_state)
            print(f"Screenshot: {observation.screenshot}")

            response = mock_policy(observation.page_state)
            action = extract_action(response)
            print(f"Action: {action.raw}")

            if action.is_terminal:
                break

            result = executor.execute(action)
            print(f"Executed: {result.success} {result.message}")
            page.wait_for_timeout(600)

        browser.close()


if __name__ == "__main__":
    main()
