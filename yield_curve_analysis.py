
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fredapi import Fred
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────
fred = Fred(api_key="b7b4ad206b7ff8f20cc635fb0bdcee34")

# ── DATA PULL ─────────────────────────────────────────────────────
maturities = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO",
    "1Y": "DGS1",   "2Y": "DGS2",   "5Y": "DGS5",
    "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30"
}

start = "2020-01-01"
end = datetime.today().strftime("%Y-%m-%d")

data = pd.DataFrame()
for label, series_id in maturities.items():
    series = fred.get_series(series_id, start, end)
    data[label] = series

data = data.dropna()
spread_2s10s = data["10Y"] - data["2Y"]

# ── SNAPSHOT DATES ────────────────────────────────────────────────
snapshots = {
    "Pre-Hike (Jan 2022)":  "2022-01-03",
    "Peak Hikes (Oct 2023)": "2023-10-02",
    "Today":                 data.index[-1].strftime("%Y-%m-%d")
}

maturity_order = ["1M","3M","6M","1Y","2Y","5Y","10Y","20Y","30Y"]

# ── PLOT ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("US Treasury Yield Curve Analysis (2020–Present)",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1 — Yield Curve Snapshots
ax1 = fig.add_subplot(gs[0, :])
colors = ["steelblue", "tomato", "green"]
for (label, date), color in zip(snapshots.items(), colors):
    nearest = data.index[data.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]]
    ax1.plot(maturity_order, data.loc[nearest, maturity_order].values,
             marker="o", label=f"{label} ({nearest.strftime('%b %d %Y')})",
             linewidth=2, color=color)
ax1.set_title("Yield Curve at Key Macro Inflection Points", fontweight="bold")
ax1.set_ylabel("Yield (%)")
ax1.set_xlabel("Maturity")
ax1.legend()
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

# Panel 2 — 10Y Yield Over Time
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(data.index, data["10Y"], color="steelblue", linewidth=1.5)
ax2.set_title("10-Year Treasury Yield Over Time", fontweight="bold")
ax2.set_ylabel("Yield (%)")
ax2.fill_between(data.index, data["10Y"], alpha=0.15, color="steelblue")

# Panel 3 — 2s10s Spread (Inversion Signal)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(spread_2s10s.index, spread_2s10s.values, color="darkred", linewidth=1.5)
ax3.axhline(0, color="black", linestyle="--", linewidth=1)
ax3.fill_between(spread_2s10s.index, spread_2s10s.values, 0,
                 where=(spread_2s10s.values < 0), color="red", alpha=0.3, label="Inversion")
ax3.fill_between(spread_2s10s.index, spread_2s10s.values, 0,
                 where=(spread_2s10s.values >= 0), color="green", alpha=0.15, label="Normal")
ax3.set_title("2s10s Spread — Yield Curve Inversion Signal", fontweight="bold")
ax3.set_ylabel("Spread (%)")
ax3.legend()

plt.savefig("yield_curve_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as yield_curve_analysis.png")