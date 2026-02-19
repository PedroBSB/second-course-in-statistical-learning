"""
export_script.py

Reads a Python file from the source/ folder, uploads it to CodeImage
(https://app.codeimage.dev) using the VSCode Dark theme and no watermark,
exports it as an SVG, and saves the result to images/<script_name>.svg.

Usage:
    python export_script.py source/linear_regression.py
    python export_script.py source/my_script.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

CODEIMAGE_URL = "https://app.codeimage.dev/"

IMAGES_DIR = Path(__file__).parent / "images"
SOURCE_DIR = Path(__file__).parent / "source"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dismiss_dialogs(page) -> None:
    """Dismiss any modal dialogs (e.g. 'What's new') using the Escape key."""
    try:
        page.wait_for_selector('[role="dialog"]', timeout=5000)
        page.keyboard.press("Escape")
        page.wait_for_selector('[role="dialog"]', state="hidden", timeout=5000)
    except PlaywrightTimeoutError:
        pass  # No dialog present – that's fine


def _get_main_editor(page):
    """Return a locator for the single editable CodeMirror content area."""
    # Only the active editor tab has contenteditable="true"
    return page.locator('.cm-content[contenteditable="true"]').first


def _set_editor_code(page, code: str) -> None:
    """Replace the CodeMirror editor content with *code* via keyboard."""
    editor = _get_main_editor(page)
    editor.click()
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.press("Delete")
    page.wait_for_timeout(200)
    # Use fill-equivalent: dispatch directly via CodeMirror JS API
    page.evaluate(
        """(code) => {
            const el = document.querySelector('.cm-content[contenteditable="true"]');
            const view = el?.cmView?.rootView;
            if (view) {
                view.dispatch({
                    changes: { from: 0, to: view.state.doc.length, insert: code },
                });
            }
        }""",
        code,
    )
    page.wait_for_timeout(300)


def _set_editor_code_via_clipboard(page, code: str) -> None:
    """Set editor content by writing to the clipboard and pasting."""
    # Write code to clipboard via JS
    page.evaluate(
        "async (text) => { await navigator.clipboard.writeText(text); }",
        code,
    )
    editor = _get_main_editor(page)
    editor.click()
    page.keyboard.press("ControlOrMeta+a")
    page.wait_for_timeout(200)
    page.keyboard.press("ControlOrMeta+v")
    page.wait_for_timeout(800)


def _set_language_python(page) -> None:
    """Select Python as the editor language."""
    page.click("#frameLanguageField-trigger")
    page.wait_for_timeout(500)
    page.get_by_role("option", name="Python", exact=True).click()
    page.wait_for_timeout(300)


def _set_theme_vscode_dark(page) -> None:
    """Select the VSCode Dark syntax theme."""
    page.click("#frameSyntaxHighlightField-trigger")
    page.wait_for_timeout(500)
    page.get_by_role("option", name="VSCode Dark", exact=True).click()
    page.wait_for_timeout(300)


def _hide_watermark(page) -> None:
    """Disable the CodeImage watermark."""
    # The watermark section has a label pointing to 'frameShowWatermarkField'.
    # The Hide button is the second tab inside the watermark segmented control.
    watermark_label = page.locator('label[for="frameShowWatermarkField"]')
    watermark_label.wait_for(timeout=5000)

    # Find the parent container and click the "Hide" tab inside it
    hide_btn = page.evaluate("""() => {
        const label = document.querySelector('label[for="frameShowWatermarkField"]');
        if (!label) return null;
        const container = label.closest('div') || label.parentElement;
        const tabs = Array.from(container?.querySelectorAll('[role="tab"]') ?? []);
        const hideTab = tabs.find(t => t.textContent?.trim() === 'Hide');
        return hideTab ? hideTab.id : null;
    }""")

    if hide_btn:
        page.click(f"#{hide_btn}")
        page.wait_for_timeout(300)
    else:
        # Fallback: look for an unchecked Hide tab in the watermark area
        page.locator('label[for="frameShowWatermarkField"] ~ div [role="tab"]:has-text("Hide")').click()
        page.wait_for_timeout(300)


def _export_svg(page, download_dir: Path) -> Path:
    """Open the export dialog, select SVG, confirm, and return the downloaded file path."""
    # Open export dialog
    page.get_by_text("Export", exact=True).click()
    page.wait_for_selector('[role="dialog"]', timeout=10000)
    page.wait_for_timeout(1000)

    # Make sure "Export as image" tab is active (it should be by default)
    page.get_by_role("tab", name="Export as image", exact=True).click()
    page.wait_for_timeout(300)

    # Select SVG format
    page.get_by_role("tab", name="SVG", exact=True).click()
    page.wait_for_timeout(300)

    # Confirm / download
    with page.expect_download(timeout=30000) as dl_info:
        page.get_by_role("button", name="Confirm", exact=True).click()

    download = dl_info.value
    dest = download_dir / download.suggested_filename
    download.save_as(str(dest))
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def export_to_svg(source_file: Path) -> Path:
    """Export *source_file* as an SVG to the images/ directory.

    Returns the path of the saved SVG file.
    """
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    code = source_file.read_text(encoding="utf-8")
    stem = source_file.stem
    output_path = IMAGES_DIR / f"{stem}.svg"

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1400, "height": 900},
            accept_downloads=True,
            permissions=["clipboard-read", "clipboard-write"],
        )
        page = context.new_page()

        print(f"  Loading CodeImage…")
        page.goto(CODEIMAGE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        print("  Dismissing dialogs…")
        _dismiss_dialogs(page)

        print("  Setting language to Python…")
        _set_language_python(page)

        print("  Setting theme to VSCode Dark…")
        _set_theme_vscode_dark(page)

        print("  Hiding watermark…")
        _hide_watermark(page)

        print("  Injecting code…")
        # Try clipboard approach first; fall back to CodeMirror dispatch
        try:
            _set_editor_code_via_clipboard(page, code)
        except Exception:
            _set_editor_code(page, code)

        page.wait_for_timeout(500)

        print("  Exporting as SVG…")
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = _export_svg(page, Path(tmp_dir))
            shutil.copy2(downloaded, output_path)

        browser.close()

    print(f"  Saved → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Python source file as a SVG code screenshot via CodeImage."
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="Path to the Python source file (default: all .py files in source/)",
    )
    args = parser.parse_args()

    if args.source:
        files = [args.source]
    else:
        files = sorted(SOURCE_DIR.glob("*.py"))
        if not files:
            print(f"No .py files found in {SOURCE_DIR}", file=sys.stderr)
            sys.exit(1)

    for source_file in files:
        print(f"\nExporting: {source_file.name}")
        try:
            out = export_to_svg(source_file)
            print(f"  Done: {out}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
