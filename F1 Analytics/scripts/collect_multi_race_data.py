import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api.openf1.org/v1"
DRIVERS = [1, 4, 11, 16, 44, 63]


SESSIONS = {
    9550: "Austria",
    9523: "Monaco",
    9590: "Monza",
    9625: "Mexico",
    9636: "Brazil",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "multi_race_raw"
ENDPOINTS = ["car_data", "location"]

REQUEST_PAUSE_SECONDS = 1.5
MAX_RETRIES = 6


def fetch(endpoint: str, params: dict) -> list:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    429 rate-limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return []


def save_json(data: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def main() -> None:
    for session_key, name in SESSIONS.items():
        print(f"=== {name} (session_key={session_key}) ===")
        for driver_number in DRIVERS:
            for endpoint in ENDPOINTS:
                out_path = RAW_DIR / str(session_key) / endpoint / f"driver_{driver_number}.json"
                if out_path.exists():
                    print(f"  [driver {driver_number}] {endpoint}: already collected, skipping")
                    continue
                data = fetch(endpoint, {"session_key": session_key, "driver_number": driver_number})
                save_json(data, out_path)
                print(f"  [driver {driver_number}] {endpoint}: {len(data)} rows -> {out_path}")
                time.sleep(REQUEST_PAUSE_SECONDS)

    print("Done.")


if __name__ == "__main__":
    main()
