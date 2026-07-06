import json
import re
from pathlib import Path

import pandas as pd

LAP_GAP_PATTERN = re.compile(r"\+\s*(\d+)\s*LAPS?", re.IGNORECASE)
ASSUMED_SECONDS_PER_LAP = 90.0


def _parse_lap_gap(value):


    if isinstance(value, str):
        match = LAP_GAP_PATTERN.match(value.strip())
        if match:
            return float(match.group(1)) * ASSUMED_SECONDS_PER_LAP
        return None
    return value

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DRIVERS = [1, 4, 11, 16, 44, 63]


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


def load_laps_for_driver(driver_number: int) -> pd.DataFrame:
    laps = load_json(RAW_DIR / "laps" / f"driver_{driver_number}.json")
    if laps.empty or "date_start" not in laps.columns:
        return pd.DataFrame()
    laps = laps.copy()
    laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601")
    return laps.sort_values("date_start").reset_index(drop=True)


def load_stints_for_driver(driver_number: int) -> pd.DataFrame:
    return load_json(RAW_DIR / "stints" / f"driver_{driver_number}.json")


def assign_lap_number_and_compound(base: pd.DataFrame, driver_number: int) -> pd.DataFrame:


    laps = load_laps_for_driver(driver_number)
    if not laps.empty:
        base = pd.merge_asof(
            base.sort_values("date"), laps[["date_start", "lap_number"]].rename(columns={"date_start": "date"}),
            on="date", direction="backward",
        )
        base["lap_number"] = base["lap_number"].fillna(1)
    else:
        base["lap_number"] = 1

    stints = load_stints_for_driver(driver_number)
    if not stints.empty:
        stints = stints.sort_values("lap_start")
        base["compound"] = None
        for _, stint in stints.iterrows():
            mask = (base["lap_number"] >= stint["lap_start"]) & (base["lap_number"] <= stint["lap_end"])
            base.loc[mask, "compound"] = stint["compound"]
        base["compound"] = base["compound"].ffill().bfill()

    return base


def build_driver_telemetry(driver_number: int) -> pd.DataFrame:
    car = load_json(RAW_DIR / "car_data" / f"driver_{driver_number}.json")
    loc = load_json(RAW_DIR / "location" / f"driver_{driver_number}.json")
    intervals = load_json(RAW_DIR / "intervals" / f"driver_{driver_number}.json")
    position = load_json(RAW_DIR / "position" / f"driver_{driver_number}.json")

    if car.empty:
        return pd.DataFrame()

    base = car[["date", "n_gear", "drs", "throttle", "rpm", "brake", "speed"]].copy()
    base = base.set_index("date")

    if not loc.empty:
        loc_idx = loc[["date", "x", "y", "z"]].set_index("date")
        base = base.join(loc_idx, how="outer")

    if not intervals.empty:
        cols = [c for c in ["gap_to_leader", "interval"] if c in intervals.columns]
        intervals = intervals.copy()
        for col in cols:
            intervals[col] = pd.to_numeric(intervals[col].map(_parse_lap_gap), errors="coerce")
        int_idx = intervals[["date"] + cols].set_index("date")
        base = base.join(int_idx, how="outer")

    if not position.empty and "position" in position.columns:
        pos_idx = position[["date", "position"]].set_index("date")
        base = base.join(pos_idx, how="outer")

    base = base.sort_index()

    numeric_cols = base.select_dtypes(include="number").columns
    base[numeric_cols] = base[numeric_cols].interpolate(method="time").ffill().bfill()

    base = base.reindex(car["date"]).reset_index()
    base = base.rename(columns={"index": "date"})
    base.insert(1, "driver_number", driver_number)

    base = assign_lap_number_and_compound(base, driver_number)
    return base


def build_event_dataset(endpoint: str) -> pd.DataFrame:
    frames = []
    for driver_number in DRIVERS:
        df = load_json(RAW_DIR / endpoint / f"driver_{driver_number}.json")
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns:
        combined = combined.sort_values(["date"]).reset_index(drop=True)
    elif "date_start" in combined.columns:
        combined = combined.sort_values(["date_start"]).reset_index(drop=True)
    return combined


def build_session_dataset(endpoint: str, session_key: int = 9550) -> pd.DataFrame:
    return load_json(RAW_DIR / endpoint / f"session_{session_key}.json")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for driver_number in DRIVERS:
        telemetry = build_driver_telemetry(driver_number)
        if telemetry.empty:
            print(f"driver {driver_number}: no car_data, skipping")
            continue
        out_path = PROCESSED_DIR / f"driver_{driver_number}_telemetry.csv"
        telemetry.to_csv(out_path, index=False)
        print(f"driver {driver_number}: {len(telemetry)} rows -> {out_path}")

    for endpoint in ["laps", "pit", "stints"]:
        df = build_event_dataset(endpoint)
        out_path = PROCESSED_DIR / f"{endpoint}.csv"
        df.to_csv(out_path, index=False)
        print(f"{endpoint}: {len(df)} rows -> {out_path}")

    for endpoint in ["weather", "race_control"]:
        df = build_session_dataset(endpoint)
        out_path = PROCESSED_DIR / f"{endpoint}.csv"
        df.to_csv(out_path, index=False)
        print(f"{endpoint}: {len(df)} rows -> {out_path}")

    drivers_df = build_session_dataset("drivers")
    drivers_df.to_csv(PROCESSED_DIR / "drivers.csv", index=False)
    print(f"drivers: {len(drivers_df)} rows -> {PROCESSED_DIR / 'drivers.csv'}")

    print("Done.")


if __name__ == "__main__":
    main()
