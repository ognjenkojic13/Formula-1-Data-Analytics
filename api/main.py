from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DRIVERS = [1, 4, 11, 16, 44, 63]
MAX_ROW_DELAY_SECONDS = 2.0
PLAYBACK_SPEEDUP = 3


RACE_START = pd.Timestamp("2024-06-30 13:03:02.157000+00:00")

app = FastAPI(title="F1 Telemetry Replay Service")


def _load_all_drivers_telemetry() -> pd.DataFrame:
    frames = []
    for driver_number in DRIVERS:
        path = PROCESSED_DIR / f"driver_{driver_number}_telemetry.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], format="ISO8601")
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["date"] >= RACE_START]
    return combined.sort_values("date").reset_index(drop=True)


def _stream_dataframe(df: pd.DataFrame, columns: list[str]):
    import time as _time

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available - run scripts/collect_data.py and scripts/combine_interpolate.py first")

    subset = df[columns].copy()
    yield ",".join(columns) + "\n"

    prev_ts = None
    for _, row in subset.iterrows():
        ts = row["date"]
        if prev_ts is not None:
            delay = min((ts - prev_ts).total_seconds() / PLAYBACK_SPEEDUP, MAX_ROW_DELAY_SECONDS)
            if delay > 0:
                _time.sleep(delay)
        prev_ts = ts
        yield ",".join(str(v) for v in row.values) + "\n"


@app.get("/stream/car_data")
def stream_car_data():
    df = _load_all_drivers_telemetry()
    columns = ["date", "driver_number", "n_gear", "drs", "throttle", "rpm", "brake", "speed"]
    return StreamingResponse(_stream_dataframe(df, columns), media_type="text/csv")


@app.get("/stream/location")
def stream_location():
    df = _load_all_drivers_telemetry()
    columns = ["date", "driver_number", "x", "y", "z"]
    return StreamingResponse(_stream_dataframe(df, columns), media_type="text/csv")


@app.get("/stream/intervals")
def stream_intervals():
    df = _load_all_drivers_telemetry()
    available = [c for c in ["gap_to_leader", "interval"] if c in df.columns]
    columns = ["date", "driver_number"] + available
    return StreamingResponse(_stream_dataframe(df, columns), media_type="text/csv")


@app.get("/stream/full")
def stream_full():


    df = _load_all_drivers_telemetry()
    columns = [c for c in [
        "date", "driver_number", "n_gear", "drs", "throttle", "rpm", "brake", "speed",
        "x", "y", "z", "gap_to_leader", "interval", "position", "lap_number", "compound",
    ] if c in df.columns]
    return StreamingResponse(_stream_dataframe(df, columns), media_type="text/csv")


@app.get("/stream/weather")
def stream_weather():
    path = PROCESSED_DIR / "weather.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="weather.csv not found")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    columns = list(df.columns)
    return StreamingResponse(_stream_dataframe(df, columns), media_type="text/csv")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
