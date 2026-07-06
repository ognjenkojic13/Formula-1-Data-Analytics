import argparse
import json
import queue
import threading
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer

STREAMING_CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "streaming_chunks"

FLUSH_INTERVAL_SECONDS = 5
FLUSH_MIN_ROWS = 1


live_queue: "queue.Queue" = queue.Queue(maxsize=2000)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:


    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", errors="coerce")
    for col in df.columns:
        if col == "date":
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == df[col].notna().sum():
            df[col] = numeric
    if "driver_number" in df.columns:
        df["driver_number"] = df["driver_number"].astype("Int64")
    return df


def _flush_buffer(topic: str, buffer: list[dict]) -> None:
    if not buffer:
        return
    out_dir = STREAMING_CHUNKS_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _coerce_types(pd.DataFrame(buffer))
    out_path = out_dir / f"part-{int(time.time() * 1000)}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  flushed {len(buffer)} rows -> {out_path}")


def consume(topics: list[str], bootstrap_servers: str, group_id: str) -> None:
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=30_000,
    )
    print(f"Subscribed to {topics} on {bootstrap_servers} (group={group_id})")

    buffers: dict[str, list[dict]] = {t: [] for t in topics}
    last_flush = time.time()
    total = 0

    for message in consumer:
        record = message.value
        buffers[message.topic].append(record)
        total += 1

        try:
            live_queue.put_nowait({"topic": message.topic, **record})
        except queue.Full:
            live_queue.get_nowait()
            live_queue.put_nowait({"topic": message.topic, **record})

        if time.time() - last_flush >= FLUSH_INTERVAL_SECONDS:
            for topic, buf in buffers.items():
                _flush_buffer(topic, buf)
                buf.clear()
            last_flush = time.time()

    for topic, buf in buffers.items():
        _flush_buffer(topic, buf)

    print(f"Consumer finished (no messages for 30s). Total consumed: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+", required=True)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--group-id", default="f1-consumer")
    args = parser.parse_args()

    consume(args.topics, args.bootstrap_servers, args.group_id)
