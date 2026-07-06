import argparse
import json
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DRIVERS = [1, 4, 11, 16, 44, 63]

DRIVER_COLORS = {
    1: "#1e3a8a",
    4: "#f97316",
    11: "#0ea5e9",
    16: "#dc2626",
    44: "#14b8a6",
    63: "#22d3ee",
}

latest: dict[int, dict] = {}
lock = threading.Lock()


def load_track_outline(reference_driver: int = 16, reference_lap: int = 30) -> pd.DataFrame:


    path = PROCESSED_DIR / f"driver_{reference_driver}_telemetry.csv"
    if not path.exists():
        return pd.DataFrame(columns=["x", "y"])
    df = pd.read_csv(path, usecols=["date", "lap_number", "x", "y"])
    lap = df[df["lap_number"] == reference_lap].sort_values("date")
    if lap.empty:
        lap = df[df["lap_number"] == df["lap_number"].median()].sort_values("date")
    return lap[["x", "y"]]


def consume_loop(topic: str, bootstrap_servers: str) -> None:
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    for message in consumer:
        record = message.value
        try:
            driver_number = int(float(record["driver_number"]))
        except (KeyError, ValueError, TypeError):
            continue
        with lock:
            latest[driver_number] = record


def _smooth_points(pts: np.ndarray, window: int = 9) -> np.ndarray:


    if window < 3 or len(pts) < window:
        return pts
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.column_stack([
        np.convolve(padded[:, 0], kernel, mode="valid"),
        np.convolve(padded[:, 1], kernel, mode="valid"),
    ])
    return smoothed[:len(pts)]


def draw_track(ax, track_df: pd.DataFrame) -> None:


    raw_pts = track_df[["x", "y"]].to_numpy(dtype=float)
    if len(raw_pts) < 2:
        return
    pts = _smooth_points(raw_pts)
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))

    segment_points = 6
    segments = [pts[i:i + 2] for i in range(len(pts) - 1)]
    colors = ["#e74c3c" if (i // segment_points) % 2 == 0 else "#f5f5f5" for i in range(len(segments))]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=34,
                                      capstyle="round", joinstyle="round", zorder=1))

    ax.plot(pts[:, 0], pts[:, 1], color="#3b3b3b", linewidth=26, solid_capstyle="round", zorder=2)

    margin = diag * 0.05
    ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax.set_ylim(pts[:, 1].min() - margin, pts[:, 1].max() + margin)


def _f(record: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="f1-full")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    threading.Thread(target=consume_loop, args=(args.topic, args.bootstrap_servers), daemon=True).start()

    track = load_track_outline()

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    ax_track = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])
    ax_panel.axis("off")

    draw_track(ax_track, track)
    ax_track.set_aspect("equal")
    ax_track.set_xticks([])
    ax_track.set_yticks([])
    dots = ax_track.scatter([], [], s=120, zorder=5)
    dot_labels = [ax_track.text(0, 0, "", fontsize=9, fontweight="bold") for _ in DRIVERS]

    title = fig.suptitle("Čekam podatke sa Kafka teme...", fontsize=12)
    panel_texts = [ax_panel.text(0.02 + 0.5 * (i % 2), 0.95 - 0.32 * (i // 2), "",
                                  fontsize=10, va="top", family="monospace",
                                  transform=ax_panel.transAxes)
                   for i in range(len(DRIVERS))]

    def update(_frame):
        with lock:
            snapshot = dict(latest)

        if not snapshot:
            return [dots, title, *dot_labels, *panel_texts]

        xs, ys, colors = [], [], []
        max_lap = 0
        for driver_number, rec in snapshot.items():
            xs.append(_f(rec, "x"))
            ys.append(_f(rec, "y"))
            colors.append(DRIVER_COLORS.get(driver_number, "#3498db"))
            max_lap = max(max_lap, int(_f(rec, "lap_number", 1)))

        dots.set_offsets(list(zip(xs, ys)))
        dots.set_color(colors)
        for lbl, (driver_number, rec) in zip(dot_labels, snapshot.items()):
            lbl.set_position((_f(rec, "x"), _f(rec, "y")))
            lbl.set_text(str(driver_number))
        for lbl in dot_labels[len(snapshot):]:
            lbl.set_text("")

        title.set_text(f"Krug {max_lap} — uživo sa Kafka teme '{args.topic}'")

        sorted_items = sorted(snapshot.items(), key=lambda kv: _f(kv[1], "gap_to_leader", 999))
        for txt, (driver_number, rec) in zip(panel_texts, sorted_items):
            compound = rec.get("compound") or "?"
            txt.set_color(DRIVER_COLORS.get(driver_number, "black"))
            txt.set_text(
                f"Vozač {driver_number}\n"
                f"  Brzina:   {_f(rec, 'speed'):.0f} km/h\n"
                f"  Gas:      {_f(rec, 'throttle'):.0f}%\n"
                f"  Kočnica:  {_f(rec, 'brake'):.0f}%\n"
                f"  Brzina prenosa: {int(_f(rec, 'n_gear'))}\n"
                f"  Gap:      {_f(rec, 'gap_to_leader'):.2f}s\n"
                f"  Interval: {_f(rec, 'interval'):.2f}s\n"
                f"  Guma:     {compound}"
            )
        for txt in panel_texts[len(sorted_items):]:
            txt.set_text("")

        return [dots, title, *dot_labels, *panel_texts]

    ani = FuncAnimation(fig, update, interval=300, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
