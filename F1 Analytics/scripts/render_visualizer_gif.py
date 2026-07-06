import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kafka"))
import visualizer as v

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
DRIVERS = [1, 4, 11, 16, 44, 63]
START_LAP, END_LAP = 29, 31
N_FRAMES = 70


def main() -> None:
    track = v.load_track_outline()

    frames_data = {}
    for d in DRIVERS:
        df = pd.read_csv(PROCESSED_DIR / f"driver_{d}_telemetry.csv")
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        window = df[(df["lap_number"] >= START_LAP) & (df["lap_number"] <= END_LAP)].reset_index(drop=True)
        frames_data[d] = window

    t_min = max(df["date"].min() for df in frames_data.values())
    t_max = min(df["date"].max() for df in frames_data.values())
    sample_times = pd.date_range(t_min, t_max, periods=N_FRAMES)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])
    ax_track = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])
    ax_panel.axis("off")

    v.draw_track(ax_track, track)
    ax_track.set_aspect("equal")
    ax_track.set_xticks([])
    ax_track.set_yticks([])
    dots = ax_track.scatter([], [], s=160, zorder=5, edgecolors="white", linewidths=1.5)
    dot_labels = [ax_track.text(0, 0, "", fontsize=9, fontweight="bold", zorder=6) for _ in DRIVERS]
    title = fig.suptitle("", fontsize=12)

    panel_texts = [ax_panel.text(0.02 + 0.5 * (i % 2), 0.95 - 0.32 * (i // 2), "",
                                  fontsize=10, va="top", family="monospace",
                                  transform=ax_panel.transAxes)
                   for i in range(len(DRIVERS))]

    def get_row(d: int, t) -> dict:
        df = frames_data[d]
        idx = (df["date"] - t).abs().idxmin()
        return df.loc[idx].to_dict()

    def update(frame_idx):
        t = sample_times[frame_idx]
        records = {d: get_row(d, t) for d in DRIVERS}

        xs = [records[d]["x"] for d in DRIVERS]
        ys = [records[d]["y"] for d in DRIVERS]
        colors = [v.DRIVER_COLORS.get(d, "#3498db") for d in DRIVERS]
        dots.set_offsets(list(zip(xs, ys)))
        dots.set_color(colors)
        for lbl, d in zip(dot_labels, DRIVERS):
            lbl.set_position((records[d]["x"], records[d]["y"]))
            lbl.set_text(str(d))

        max_lap = max(int(records[d]["lap_number"]) for d in DRIVERS)
        title.set_text(f"Krug {max_lap} — Austrija 2024")

        sorted_drivers = sorted(DRIVERS, key=lambda d: v._f(records[d], "gap_to_leader", 999))
        for txt, d in zip(panel_texts, sorted_drivers):
            rec = records[d]
            txt.set_color(v.DRIVER_COLORS.get(d, "black"))
            txt.set_text(
                f"Vozač {d}\n"
                f"  Brzina:   {v._f(rec, 'speed'):.0f} km/h\n"
                f"  Gas:      {v._f(rec, 'throttle'):.0f}%\n"
                f"  Kočnica:  {v._f(rec, 'brake'):.0f}%\n"
                f"  Gap:      {v._f(rec, 'gap_to_leader'):.2f}s\n"
                f"  Guma:     {rec.get('compound') or '?'}"
            )
        return [dots, title, *dot_labels, *panel_texts]

    ani = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    plt.tight_layout()

    out_path = ASSETS_DIR / "live_visualizer.gif"
    ani.save(out_path, writer=PillowWriter(fps=8), dpi=90)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
