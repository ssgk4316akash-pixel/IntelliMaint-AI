from pathlib import Path
import subprocess
import sys
import threading
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"


def run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def open_browser_when_ready() -> None:
    url = "http://127.0.0.1:8000"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{url}/api/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, TimeoutError, URLError):
            pass
        time.sleep(0.25)
    print(f"\nBackend did not become ready within 60 seconds. Open {url} manually.")


def main() -> None:
    print("Building frontend...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    run([npm, "run", "build"], FRONTEND)

    print("\nStarting IntelliMaint AI...")
    print("The browser will open after the backend is ready.")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run(
        "app.main:app",
        app_dir=str(BACKEND),
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
