"""Generates Graphs/hir_formula.png — v stilu originalnega AHI.png"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "Graphs" / "hir_formula.png"
OUT_PATH_AHI = PROJECT_ROOT / "Graphs" / "ahi_formula.png"

fig, ax = plt.subplots(figsize=(12, 3.2), facecolor="#111111")
ax.set_facecolor("#111111")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

formula = (
    r"$AHI = \sqrt{\dfrac{"
    r"w_k \cdot K^2 \;+\; w_r \cdot R^2 \;+\; w_a \cdot A^2 \;+\; w_c \cdot C^2"
    r"}{\sum w}} \;\times\; 100$"
)

ax.text(
    0.5, 0.62,
    formula,
    ha="center", va="center",
    fontsize=26,
    color="white",
    transform=ax.transAxes,
)

legend = (
    r"$K$ = sklearn failure prob $\;(w_k{=}0.30)$"
    r"$\quad|\quad$"
    r"$R$ = TF bottleneck clf prob $\;(w_r{=}0.40)$"
    r"$\quad|\quad$"
    r"$A$ = anomaly score $\;(w_a{=}0.20)$"
    r"$\quad|\quad$"
    r"$C$ = cluster risk score $\;(w_c{=}0.10)$"
)

ax.text(
    0.5, 0.18,
    legend,
    ha="center", va="center",
    fontsize=9.5,
    color="#aaaaaa",
    transform=ax.transAxes,
)

plt.tight_layout(pad=0.4)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="#111111")
plt.savefig(OUT_PATH_AHI, dpi=180, bbox_inches="tight", facecolor="#111111")
plt.close()
print(f"Shranjeno: {OUT_PATH}")
print(f"Shranjeno: {OUT_PATH_AHI}")
