from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "streaming_chunks"
DRIVERS = [1, 4, 11, 16, 44, 63]
CHUNK_SECONDS = 5
TOPIC_NAME = "f1-car-data"


def main() -> None:
    frames = []
    for d in DRIVERS:
        path = PROCESSED_DIR / f"driver_{d}_telemetry.csv"
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        frames.append(df[["date", "driver_number", "speed", "throttle", "brake", "rpm", "n_gear", "gap_to_leader"]])

    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    combined["gap_to_leader"] = pd.to_numeric(combined["gap_to_leader"], errors="coerce")

    out_dir = CHUNKS_DIR / TOPIC_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.parquet"):
        f.unlink()

    start = combined["date"].iloc[0]
    combined["chunk_id"] = ((combined["date"] - start).dt.total_seconds() // CHUNK_SECONDS).astype(int)

    for chunk_id, group in combined.groupby("chunk_id"):
        out_path = out_dir / f"part-{chunk_id:06d}.parquet"
        group.drop(columns=["chunk_id"]).to_parquet(out_path, index=False)

    n_chunks = combined["chunk_id"].nunique()
    print(f"Wrote {n_chunks} chunk files ({CHUNK_SECONDS}s each) to {out_dir}")


if __name__ == "__main__":
    main()
