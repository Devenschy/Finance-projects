

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

# ── CONFIGURATION ────────────────────────────────────────────────
tickers = ["JPM", "GS", "MS", "BAC", "C"]
start = "2022-01-01"
end = datetime.today().strftime("%Y-%m-%d")
rf = 0.05  # risk-free rate (approximate current T-bill yield)

# ── DATA PULL ─────────────────────────────────────────────────────
raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
returns = raw.pct_change().dropna()

# ── METRICS ───────────────────────────────────────────────────────
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = (ann_return - rf) / ann_vol
max_dd = ((raw / raw.cummax()) - 1).min()
cumulative = (1 + returns).cumprod()

summary = pd.DataFrame({
    "Annualized Return": ann_return.map("{:.1%}".format),
    "Annualized Volatility": ann_vol.map("{:.1%}".format),
    "Sharpe Ratio": sharpe.map("{:.2f}".format),
    "Max Drawdown": max_dd.map("{:.1%}".format)
})

print("\n── RISK/RETURN SUMMARY ──────────────────────────────")
print(summary.to_string())

# ── PLOT ──────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Equity Risk & Return Analysis — Major US Banks (2022–Present)",
             fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel 1 — Cumulative Returns
ax1 = fig.add_subplot(gs[0, :])
for ticker in tickers:
    ax1.plot(cumulative.index, cumulative[ticker], label=ticker, linewidth=1.8)
ax1.set_title("Cumulative Returns", fontweight="bold")
ax1.set_ylabel("Growth of $1")
ax1.legend(loc="upper left")
ax1.axhline(1, color="black", linestyle="--", linewidth=1)