"""논문 그림 생성. 수치는 analyze_national.py / analyze_president.py 실행 출력에서 채록.

  fig1_sideband.png — 지선 전국 ρ(d) 범프헌트 (관측·시뮬·선형외삽·d=0)
  fig2_ratio.png    — 셸 비 r(d): 지선 vs 대선
  fig3_types.png    — 유형 분해: 광주전남 vs 대선 (기대 대 관측)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from analyze_national import fit_sideband  # noqa: E402

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# analyze_national.py (광주전남 합본, 2026-06-10 실행) / analyze_president.py
LOCAL_OBS = np.array([9, 35, 48, 72, 130, 146, 135, 191, 198, 219, 238, 301, 315, 324, 350, 362])
LOCAL_SIM = np.array([3.24, 25.82, 51.19, 76.54, 102.09, 128.53, 153.48, 178.19,
                      203.63, 229.43, 252.16, 275.99, 299.64, 321.96, 346.10, 368.51])
PRES_OBS = np.array([7, 45, 99, 136, 196, 273, 320, 330, 410, 438, 513, 552, 599, 619, 659, 722])
PRES_SIM = np.array([5.96, 48.56, 96.52, 143.85, 191.01, 240.22, 286.15, 333.75,
                     377.81, 424.36, 470.38, 514.22, 556.00, 600.70, 641.47, 683.28])
D = np.arange(16)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def rho(n):
    lat = np.where(D == 0, 1.0, 8.0 * D)
    return n / lat


def rho_err(n):
    lat = np.where(D == 0, 1.0, 8.0 * D)
    return np.sqrt(np.maximum(n, 1)) / lat


# ── Fig 1: bump hunt ──────────────────────────────────────────────
a, se, theta = fit_sideband(LOCAL_OBS)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.errorbar(D[1:], rho(LOCAL_OBS)[1:], yerr=rho_err(LOCAL_OBS)[1:], fmt="o",
            color="#1f4e79", ms=5, capsize=3, label="Observed  $\\rho(d)=n(d)/8d$")
ax.plot(D[1:], rho(LOCAL_SIM)[1:], "s--", color="#999999", ms=4, lw=1,
        label="Generative model")
dd = np.linspace(0, 15, 100)
ax.plot(dd, np.polyval(theta[::-1], dd), "-", color="#c0504d", lw=1.5,
        label=f"Sideband fit (d$\\geq$1) $\\to$ $\\hat\\rho(0)$={a:.2f}$\\pm${se:.2f}")
ax.errorbar([0], [LOCAL_OBS[0]], yerr=[3.0], fmt="*", color="#c0504d", ms=16,
            capsize=3, label=f"Observed exact ties: {LOCAL_OBS[0]}")
ax.annotate("", xy=(0.15, LOCAL_OBS[0] - 0.4), xytext=(0.15, a + 0.25),
            arrowprops=dict(arrowstyle="<->", color="#c0504d", lw=1))
ax.text(0.45, (a + LOCAL_OBS[0]) / 2, "excess\n(P$\\approx$0.9%)", fontsize=9,
        color="#c0504d", va="center")
ax.set_xlabel("Chebyshev distance  $d=\\max(|\\Delta A|,|\\Delta B|)$")
ax.set_ylabel("Pair density per lattice site")
ax.set_title("Near-pair shells and the $d=0$ atom\n(2026 local elections, nationwide)",
             fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 10)
fig.tight_layout()
fig.savefig(FIGS / "fig1_sideband.png", dpi=200)

# ── Fig 2: r(d) two elections ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axhline(1, color="#bbbbbb", lw=1)
ax.plot(D, LOCAL_OBS / LOCAL_SIM, "o-", color="#1f4e79", ms=5,
        label="2026 local elections")
ax.plot(D, PRES_OBS / PRES_SIM, "s-", color="#70ad47", ms=5, mfc="white",
        label="2025 presidential election")
ax.annotate("local $d=0$: 2.78", xy=(0, LOCAL_OBS[0] / LOCAL_SIM[0]),
            xytext=(1.3, 2.6), fontsize=9, color="#1f4e79",
            arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=0.8))
ax.set_xlabel("Chebyshev distance $d$")
ax.set_ylabel("Observed / model shell count  $r(d)$")
ax.set_title("The excess is confined to exact ties,\nand only in the local elections",
             fontsize=11)
ax.set_ylim(0, 3.1)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "fig2_ratio.png", dpi=200)

# ── Fig 3: type decomposition ────────────────────────────────────
types = ["Day·Day", "Mixed", "Early·Early"]
panels = [("Gwangju–Jeonnam block (local)", [0.575, 0.481, 0.421], [0, 0, 5]),
          ("Presidential, nationwide", [2.780, 0.863, 2.347], [6, 1, 0])]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=False)
x = np.arange(3)
for ax, (title, exp, obs) in zip(axes, panels):
    ax.bar(x - 0.18, exp, 0.36, color="#999999", label="Model expected")
    ax.bar(x + 0.18, obs, 0.36, color="#c0504d", label="Observed")
    ax.set_xticks(x, types)
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(obs):
        ax.text(i + 0.18, v + 0.08, str(v), ha="center", fontsize=9, color="#c0504d")
axes[0].set_ylabel("Twin pairs")
axes[0].legend(frameon=False, fontsize=9)
axes[0].text(0.9, 3.9, "all 5 in the type with\nthe smallest expectation", fontsize=8.5,
             ha="center", color="#c0504d")
axes[0].annotate("", xy=(1.95, 3.5), xytext=(1.45, 3.85),
                 arrowprops=dict(arrowstyle="->", color="#c0504d", lw=0.8))
fig.tight_layout()
fig.savefig(FIGS / "fig3_types.png", dpi=200)
print("saved:", sorted(p.name for p in FIGS.glob("*.png")))
