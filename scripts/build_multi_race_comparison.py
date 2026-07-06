import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "multi_race_raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "multi_race_processed"
DRIVERS = [1, 4, 11, 16, 44, 63]

SESSIONS = {
    9550: "Austria",
    9523: "Monaco",
    9590: "Monza",
    9625: "Mexico",
    9636: "Brazil",
}


def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def build_speed_summary() -> pd.DataFrame:
    rows = []
    for session_key, name in SESSIONS.items():
        for driver_number in DRIVERS:
            car = load_json(RAW_DIR / str(session_key) / "car_data" / f"driver_{driver_number}.json")
            if car.empty:
                continue
            moving = car[car["speed"] > 0]
            rows.append({
                "session_key": session_key,
                "track": name,
                "driver_number": driver_number,
                "avg_speed": moving["speed"].mean(),
                "max_speed": moving["speed"].max(),
            })
    return pd.DataFrame(rows)


def build_track_shape(session_key: int, name: str) -> pd.DataFrame:
    for driver_number in DRIVERS:
        car = load_json(RAW_DIR / str(session_key) / "car_data" / f"driver_{driver_number}.json")
        loc = load_json(RAW_DIR / str(session_key) / "location" / f"driver_{driver_number}.json")
        if car.empty or loc.empty:
            continue

        merged = pd.merge_asof(
            loc[["date", "x", "y"]], car[["date", "speed"]], on="date", direction="nearest",
        )
        fast = merged[merged["speed"] > 80].reset_index(drop=True)
        if len(fast) < 500:
            continue


        ONE_LAP_ROWS = 320
        mid = len(fast) // 2
        start = max(0, mid - ONE_LAP_ROWS // 2)
        end = start + ONE_LAP_ROWS
        return fast.iloc[start:end][["x", "y"]].reset_index(drop=True)

    return pd.DataFrame(columns=["x", "y"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_speed_summary()
    summary.to_csv(OUT_DIR / "speed_summary.csv", index=False)
    print(f"speed_summary: {len(summary)} rows -> {OUT_DIR / 'speed_summary.csv'}")

    for session_key, name in SESSIONS.items():
        shape = build_track_shape(session_key, name)
        out_path = OUT_DIR / f"track_shape_{name.lower()}.csv"
        shape.to_csv(out_path, index=False)
        print(f"track_shape ({name}): {len(shape)} rows -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
