"""Regenerates the README screenshots in assets/ from the offline demo.

    pip install playwright            # uses your installed Chrome; no browser download
    streamlit run app.py --server.headless true --server.port 8501
    python scripts/capture_screenshots.py http://localhost:8501 assets

Clicks "Try the demo vault", waits for the preview run, then captures the Run,
Results, Diffs and Graph views at 1440x900 (2x).
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]


def main(url: str = "http://localhost:8501", out: str = "assets") -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="dark")
        page.goto(url)
        page.get_by_role("button", name="Try the demo vault").click()
        page.get_by_text("Done. See the Results").wait_for(timeout=60000)
        page.wait_for_timeout(1500)

        # Run tab: expand the finished stage tracker and scroll it into view.
        page.get_by_text("Finished in").first.click()
        page.wait_for_timeout(1000)
        page.get_by_text("Done. See the Results").scroll_into_view_if_needed()
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(800)
        page.screenshot(path=str(out_dir / "demo-run.png"))
        page.mouse.wheel(0, -5000)
        page.wait_for_timeout(500)

        page.get_by_role("tab", name="Results").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out_dir / "demo-results.png"))

        page.get_by_role("tab", name="Diffs").click()
        page.wait_for_timeout(1000)
        page.get_by_text("CAP Theorem.md").first.click()
        page.get_by_role("option", name="Kubernetes.md").click()
        page.wait_for_timeout(1000)
        page.get_by_text("of 15 notes would change").click()  # close the dropdown
        page.wait_for_timeout(800)
        page.screenshot(path=str(out_dir / "demo-diff.png"))

        page.get_by_role("tab", name="Graph").click()
        page.wait_for_timeout(6000)  # let the force layout settle
        page.screenshot(path=str(out_dir / "demo-graph.png"))
        browser.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
