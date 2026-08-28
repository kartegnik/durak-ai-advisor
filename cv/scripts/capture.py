"""Capture screenshots from the Durak game window."""
import subprocess
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw_screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def capture_screenshot(filename: str | None = None) -> str:
    """Take a screenshot of the entire screen."""
    if filename is None:
        filename = f"screenshot_{int(time.time())}.png"
    filepath = OUTPUT_DIR / filename

    # Use scrot or import (ImageMagick) for screenshots
    try:
        subprocess.run(["scrot", str(filepath)], check=True)
    except FileNotFoundError:
        subprocess.run(["import", "-window", "root", str(filepath)], check=True)

    print(f"Saved: {filepath}")
    return str(filepath)


def get_next_number() -> int:
    """Find the next available screenshot number."""
    existing = list(OUTPUT_DIR.glob("game_*.png"))
    if not existing:
        return 0
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            pass
    return max(numbers) + 1 if numbers else 0


def capture_loop(interval: float = 2.0):
    """Continuously capture screenshots at given interval."""
    print(f"Capturing screenshots every {interval}s to {OUTPUT_DIR}")
    print("Press Ctrl+C to stop")

    i = get_next_number()
    print(f"Starting from game_{i:04d}.png")

    count = 0
    try:
        while True:
            capture_screenshot(f"game_{i:04d}.png")
            i += 1
            count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nCaptured {count} screenshots (game_{i-count:04d} to game_{i-1:04d})")


if __name__ == "__main__":
    capture_loop()
