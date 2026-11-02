"""Figure 2 and Figure S1, drawn from results/val_all.json and results/val_bias.json ONLY."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sst.download import ROOT

RES = ROOT / "results"
STYLE = {  # Okabe-Ito colours (colour-blind safe) + distinct markers
    "persistence": ("#999999", "o", "Persistence"), "climatology": ("#000000", "s", "Climatology"),
    "anomaly_persistence": ("#0072B2", "^", "Anomaly persistence"), "damped_persistence": ("#56B4E9", "v", "Damped persistence"),
    "ocn": ("#E69F00", "D", "OCN (10 yr)"), "ridge": ("#D55E00", "P", "Ridge"),
}


def fig2():
    d = json.loads((RES / "val_all.json").read_text())
    leads, regions = d["leads_days"], list(d["regions"])
    fig, axes = plt.subplots(len(regions), 2, figsize=(11, 4.2 * len(regions)), squeeze=False)
    for i, reg in enumerate(regions):
        for m in d["methods"]:
            c, mk, lab = STYLE[m]
            rm = [d["regions"][reg][str(h)]["methods"][m]["overall"]["rmse"] for h in leads]
            sk = [d["regions"][reg][str(h)]["skill"][m]["rmse_skill_vs_anomaly_persistence"] for h in leads]
            axes[i, 0].plot(leads, rm, color=c, marker=mk, label=lab, lw=1.6)
            axes[i, 1].plot(leads, sk, color=c, marker=mk, label=lab, lw=1.6)
        axes[i, 1].axhline(0, color="k", lw=0.6, ls=":")
        axes[i, 0].set_ylabel("RMSE (°C)")
        axes[i, 1].set_ylabel("Skill vs anomaly persistence\n(1 − RMSE/RMSE$_{ap}$, unitless)")
        axes[i, 0].set_title(f"{reg}: RMSE vs lead", fontsize=10)
        axes[i, 1].set_title(f"{reg}: skill vs anomaly persistence", fontsize=10)
        for ax in axes[i]:
            ax.set_xscale("log")
            ax.set_xticks(leads, [str(h) for h in leads])
            ax.set_xlabel("Lead time (days)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.text(0.01, 0.005, "validation 2016–2019 (1461 days per lead); all models use the 1982–2015 climatology / training years",
             fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = RES / "figures" / "fig2_val_leads.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figS1():
    d = json.loads((RES / "val_bias.json").read_text())["area_mean_anomaly"]
    fig, axes = plt.subplots(len(d), 1, figsize=(11, 3.6 * len(d)), sharex=True)
    for ax, (reg, a) in zip(np.atleast_1d(axes), d.items()):
        t = pd.to_datetime(a["dates"])
        ax.plot(t, a["daily"], color="#999999", lw=0.5, label="daily")
        yrs = sorted(a["yearly_mean"], key=int)
        ax.step([pd.Timestamp(f"{y}-01-01") for y in yrs] + [pd.Timestamp("2020-01-01")],
                [a["yearly_mean"][y] for y in yrs] + [a["yearly_mean"][yrs[-1]]], where="post", color="#D55E00", lw=2,
                label="yearly mean")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(pd.Timestamp(a["val_start"]), color="#0072B2", ls="--", lw=1.4)
        ax.axvspan(pd.Timestamp(a["val_start"]), pd.Timestamp("2020-01-01"), color="#0072B2", alpha=0.08)
        ax.text(pd.Timestamp(a["val_start"]), ax.get_ylim()[1], " validation →", va="top", fontsize=8, color="#0072B2")
        ax.text(pd.Timestamp("1982-06-01"), ax.get_ylim()[1], f"training mean {a['mean_train']:+.2f} °C | validation mean "
                f"{a['mean_val']:+.2f} °C", va="top", fontsize=8)
        ax.set_ylabel("Area-mean anomaly (°C)")
        ax.set_title(f"{reg}: area-mean SST anomaly vs 1982–2015 climatology", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Year")
    np.atleast_1d(axes)[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = RES / "figures" / "figS1_area_mean_anomaly.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig2(), figS1(), sep="\n")
