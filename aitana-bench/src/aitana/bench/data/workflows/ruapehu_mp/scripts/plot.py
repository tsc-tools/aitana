"""
Plot RSAM and its matrix-profile anomaly index for the alert message.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL_BAND = "#f0efec"
SERIES = "#2a78d6"
HIGHLIGHT = "#cde2fb"

with open(snakemake.input.alert) as fh:
    alert = json.load(fh)
df = pd.read_csv(snakemake.input.index, index_col=0, parse_dates=True)
score_start = pd.Timestamp(alert["score_start"])
end = pd.Timestamp(alert["end"])
recent = df[score_start - pd.Timedelta(hours=alert["window_hours"]) :]

plt.rcParams.update(
    {
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(10, 7.5), facecolor=SURFACE)
grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.45)
ax_context = fig.add_subplot(grid[0])
ax_rsam = fig.add_subplot(grid[1])
ax_index = fig.add_subplot(grid[2], sharex=ax_rsam)

for ax in (ax_context, ax_rsam, ax_index):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

# Context: the history the scored day is compared against.
ax_context.axvspan(score_start, end, color=NEUTRAL_BAND, linewidth=0)
ax_context.plot(df.index, df["rsam"], color=SERIES, linewidth=1)
ax_context.set_yscale("log")
ax_context.set_ylabel("RSAM (nm/s)")
ax_context.set_title(
    f"History: {alert['history_start'][:10]} to {alert['score_start'][:10]}",
    loc="left",
    fontsize=9,
    color=INK_SECONDARY,
)
ax_context.text(
    score_start, 1.0, " scored", transform=ax_context.get_xaxis_transform(),
    va="top", color=INK_SECONDARY,
)

# Detail: the scored day, with the subsequence that produced the peak.
if alert["peak_time"] is not None:
    peak_time = pd.Timestamp(alert["peak_time"])
    window_start = peak_time - pd.Timedelta(hours=alert["window_hours"]) + pd.Timedelta(
        "10min"
    )
    ax_rsam.axvspan(window_start, peak_time + pd.Timedelta("10min"), color=HIGHLIGHT,
                    linewidth=0)
    ax_rsam.text(
        window_start, 1.0, f" window with the highest index ({alert['window_hours']:g} h)",
        transform=ax_rsam.get_xaxis_transform(), va="top", color=INK_SECONDARY,
    )
ax_rsam.plot(recent.index, recent["rsam"], color=SERIES, linewidth=1.5)
ax_rsam.set_yscale("log")
ax_rsam.set_ylabel("RSAM (nm/s)")
ax_rsam.set_title("Scored period", loc="left", fontsize=9, color=INK_SECONDARY)

# Anomaly index with the alarm threshold and the peak.
ax_index.plot(recent.index, recent["anomaly_index"], color=SERIES, linewidth=1.5)
ax_index.axhline(alert["threshold"], color=INK_SECONDARY, linewidth=1, linestyle="--")
ax_index.annotate(
    f"alarm threshold {alert['threshold']:g}", (recent.index[0], alert["threshold"]),
    xytext=(4, -4), textcoords="offset points", va="top", color=INK_SECONDARY,
)
if alert["peak_time"] is not None:
    ax_index.plot(
        peak_time, alert["peak_index"], marker="o", markersize=8,
        markerfacecolor=SERIES, markeredgecolor=SURFACE, markeredgewidth=2,
    )
    ax_index.annotate(
        f"peak {alert['peak_index']:.4f} at {peak_time:%H:%M} UTC",
        (peak_time, alert["peak_index"]), xytext=(-8, -16), textcoords="offset points",
        ha="right", color=INK,
    )
ax_index.set_ylim(0, 1.05)
ax_index.set_ylabel("Anomaly index")
ax_index.set_title("Matrix-profile anomaly index", loc="left", fontsize=9,
                   color=INK_SECONDARY)
for ax in (ax_context, ax_index):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax_index.set_xlabel("UTC")
plt.setp(ax_rsam.get_xticklabels(), visible=False)
ax_rsam.xaxis.get_offset_text().set_visible(False)

status = "ALARM" if alert["alarm"] else "no alarm"
fig.suptitle(
    f"{alert['volcano']} {alert['seed_id']} RSAM: {status}, scored until "
    f"{end:%Y-%m-%d %H:%M} UTC",
    x=0.07, ha="left", fontsize=11, color=INK,
)
fig.savefig(snakemake.output[0], dpi=150, facecolor=SURFACE, bbox_inches="tight")
