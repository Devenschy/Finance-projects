import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fredapi import Fred
from datetime import datetime

# ── CONFIGURATION ─────────────────────────────────────────────────
fred = Fred(api_key="")
start = "2020-01-01"
end = datetime.today().strftime("%Y-%m-%d")

# ── DATA PULL ─────────────────────────────────────────────────────
# Commodities & macro via yfinance
tickers = {
    "WTI Oil":      "CL=F",
    "Gold":         "GC=F",
    "Copper":       "HG=F",
    "Natural Gas":  "NG=F",
    "S&P 500":      "^GSPC",
    "USD Index":    "DX-Y.NYB"
}

raw = yf.download(list(tickers.values()), start=start, end=end,
                  auto_adjust=True)["Close"]
raw.columns = list(tickers.keys())
raw = raw.ffill().dropna()

# CPI from FRED
cpi = fred.get_series("CPIAUCSL", start, end).resample("D").ffill()
cpi.name = "CPI"

# Normalize everything to 100 at start
normalized = (raw / raw.iloc[0]) * 100

# Returns for correlation
returns = raw.pct_change().dropna()

# Correlation matrix (commodities vs macro)
corr = returns[["WTI Oil","Gold","Copper","Natural Gas","S&P 500","USD Index"]].corr()

# 90-day rolling correlation: Oil vs S&P 500
roll_corr_oil_sp = returns["WTI Oil"].rolling(90).corr(returns["S&P 500"])

# Momentum signal: 3-month return for each commodity
momentum = raw[["WTI Oil","Gold","Copper","Natural Gas"]].pct_change(63) * 100

print("\n── CORRELATION MATRIX ───────────────────────────────")
print(corr.round(2).to_string())

print("\n── CURRENT 3-MONTH MOMENTUM (%) ─────────────────────")
print(momentum.iloc[-1].round(2).to_string())

# ── PLOT ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Commodity & Macro Correlation Analysis (2020–Present)",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Normalized commodity prices
ax1 = fig.add_subplot(gs[0, :])
colors = ["#e67e22","#f1c40f","#e74c3c","#3498db"]
commodities = ["WTI Oil","Gold","Copper","Natural Gas"]
for c, col in zip(commodities, colors):
    ax1.plot(normalized.index, normalized[c], label=c, linewidth=1.8, color=col)

# Mark key macro events
events = {
    "COVID Crash\n(Mar 2020)": "2020-03-23",
    "Inflation Peak\n(Jun 2022)": "2022-06-10",
    "Fed Pivot\n(Nov 2023)":  "2023-11-01"
}
for label, date in events.items():
    ts = pd.Timestamp(date)
    if ts in normalized.index:
        ax1.axvline(ts, color="gray", linestyle="--", linewidth=1)
        ax1.text(ts, ax1.get_ylim()[1]*0.92, label, fontsize=7.5,
                 ha="center", color="gray")

ax1.set_title("Commodity Prices — Normalized to 100 at Jan 2020", fontweight="bold")
ax1.set_ylabel("Indexed Price (Jan 2020 = 100)")
ax1.legend(loc="upper left")
ax1.axhline(100, color="gray", linestyle=":", linewidth=0.8)

# Panel 2 — Correlation heatmap
ax2 = fig.add_subplot(gs[1, 0])
im = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))
ax2.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=8)
ax2.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax2.text(j, i, f"{corr.values[i,j]:.2f}", ha="center",
                 va="center", fontsize=8, fontweight="bold")
ax2.set_title("Cross-Asset Correlation Matrix", fontweight="bold")
fig.colorbar(im, ax=ax2, shrink=0.8)

# Panel 3 — Rolling Oil vs S&P correlation
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(roll_corr_oil_sp.index, roll_corr_oil_sp.values,
         color="#e67e22", linewidth=1.5)
ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax3.fill_between(roll_corr_oil_sp.index, roll_corr_oil_sp.values, 0,
                 where=(roll_corr_oil_sp.values > 0),
                 color="#e67e22", alpha=0.2, label="Positive")
ax3.fill_between(roll_corr_oil_sp.index, roll_corr_oil_sp.values, 0,
                 where=(roll_corr_oil_sp.values <= 0),
                 color="#3498db", alpha=0.2, label="Negative")
ax3.set_title("Rolling 90-Day Correlation: WTI Oil vs S&P 500", fontweight="bold")
ax3.set_ylabel("Correlation")
ax3.legend()

# Panel 4 — 3-month momentum bar chart
ax4 = fig.add_subplot(gs[2, :])
mom_latest = momentum.iloc[-1]
bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in mom_latest.values]
bars = ax4.bar(mom_latest.index, mom_latest.values, color=bar_colors, width=0.5)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("3-Month Price Momentum by Commodity (%)", fontweight="bold")
ax4.set_ylabel("Return (%)")
for bar, val in zip(bars, mom_latest.values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (0.3 if val >= 0 else -1.2),
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.savefig("commodity_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as commodity_analysis.png")
