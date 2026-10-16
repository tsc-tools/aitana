import matplotlib.pyplot as plt
import pandas as pd

from aitana.visualise import set_style

set_style()

df = pd.read_csv(
    snakemake.input[0],
    skiprows=8,
    header=None,
    names=["year", "publications"],
)

df = df.sort_values("year")

fig, ax = plt.subplots(figsize=(22, 6))
ax.bar(df["year"], df["publications"])
ax.set_xlabel("Year", fontsize=18)
ax.set_ylabel("Number of Publications", fontsize=18)
ax.tick_params(axis="both", labelsize=16)
ax.grid(True, color=plt.rcParams["grid.color"], linewidth=0.8)
ax.spines[["left", "bottom"]].set_color(plt.rcParams["axes.edgecolor"])
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=plt.rcParams["xtick.color"])
plt.tight_layout()
plt.savefig(snakemake.output[0], dpi=300)
