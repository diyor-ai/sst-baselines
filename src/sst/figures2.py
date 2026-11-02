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


SKILL_YLIM = (-0.8, 0.3)


def fig2(src="val_all.json", out_name="fig2_val_leads.png",
         footnote="validation 2016–2019 (1461 days per lead); all models use the 1982–2015 climatology / training years"):
    """RMSE (left) and skill vs anomaly persistence (right). In the skill panel the y-axis is limited to
    SKILL_YLIM; values below it (climatology / OCN at short leads) are drawn as off-scale arrows with their value."""
    d = json.loads((RES / src).read_text())
    leads, regions = d["leads_days"], list(d["regions"])
    lo, hi = SKILL_YLIM
    fig, axes = plt.subplots(len(regions), 2, figsize=(11, 4.2 * len(regions)), squeeze=False)
    for i, reg in enumerate(regions):
        off = {h: [] for h in leads}
        for m in d["methods"]:
            c, mk, lab = STYLE[m]
            rm = [d["regions"][reg][str(h)]["methods"][m]["overall"]["rmse"] for h in leads]
            sk = [d["regions"][reg][str(h)]["skill"][m]["rmse_skill_vs_anomaly_persistence"] for h in leads]
            axes[i, 0].plot(leads, rm, color=c, marker=mk, label=lab, lw=1.6)
            inside = [(h, v) for h, v in zip(leads, sk) if v >= lo]
            axes[i, 1].plot([h for h, _ in inside], [v for _, v in inside], color=c, marker=mk, label=lab, lw=1.6)
            for h, v in zip(leads, sk):
                if v < lo:
                    off[h].append((m, v))
        lines = []
        for h, items in off.items():  # off-scale values: arrows at the bottom edge + one text box with the values
            for m, v in items:
                axes[i, 1].plot([h], [lo + 0.04], marker="v", color=STYLE[m][0], ms=7, ls="none", clip_on=False, zorder=6)
            if items:
                lines.append(f"lead {h} d: " + ", ".join(f"{STYLE[m][2]} {v:+.2f}" for m, v in items))
        if lines:
            axes[i, 1].text(0.02, 0.03, "off scale (▼):\n" + "\n".join(lines), transform=axes[i, 1].transAxes,
                            fontsize=7, va="bottom", ha="left", zorder=3, bbox=dict(fc="white", ec="0.7", lw=0.5))
        axes[i, 1].set_ylim(lo, hi)
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
    fig.text(0.01, 0.005, footnote, fontsize=8, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = RES / "figures" / out_name
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


def fig3(src="test_all.json", out_name="fig3_test_seasons.png", leads=(7, 30)):
    """Skill vs anomaly persistence by season. Large marker + bar = mean ± std over the calendar years,
    small dots = the individual years (each dot: that season in that year, 'DJF' = Jan, Feb, Dec of the year)."""
    d = json.loads((RES / src).read_text())
    regions, seasons = list(d["regions"]), ["DJF", "MAM", "JJA", "SON"]
    ms = [m for m in d["methods"] if m != "anomaly_persistence"]  # the reference itself is 0 by definition
    fig, axes = plt.subplots(len(regions), len(leads), figsize=(6 * len(leads), 4.3 * len(regions)), squeeze=False)
    off = np.linspace(-0.3, 0.3, len(ms))
    for i, reg in enumerate(regions):
        for j, h in enumerate(leads):
            ax = axes[i, j]
            sk = d["regions"][reg][str(h)]["skill"]
            for k, m in enumerate(ms):
                c, mk, lab = STYLE[m]
                vals = sk[m]["rmse_skill_vs_anomaly_persistence_by_season_year"]
                mean, sd = [], []
                for si, se in enumerate(seasons):
                    v = np.array([x for g, x in vals.items() if g.startswith(se + "|")])
                    mean.append(v.mean()); sd.append(v.std(ddof=1))
                    ax.plot(np.full(len(v), si + off[k]), v, ls="none", marker=".", ms=4, color=c, alpha=0.45)
                ax.errorbar(np.arange(4) + off[k], mean, yerr=sd, ls="none", marker=mk, ms=7, color=c, capsize=2,
                            label=lab, lw=1.2)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xticks(range(4), seasons)
            ax.set_ylabel("Skill vs anomaly persistence (unitless)")
            ax.set_xlabel("Season of the target date")
            ax.set_title(f"{reg}, lead {h} days", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7, ncol=2)
    p0, p1 = d["period"]["evaluated"]
    fig.text(0.01, 0.005, f"test {p0[:4]}–{p1[:4]}: marker = mean ± std over the {p1[:4] and int(p1[:4]) - int(p0[:4]) + 1} years, "
             "dots = individual years; values below 0 are worse than anomaly persistence", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = RES / "figures" / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def test_figs():
    d = json.loads((RES / "test_all.json").read_text())
    n = d["regions"][next(iter(d["regions"]))]["1"]["pairs"]["used"]
    p0, p1 = d["period"]["evaluated"]
    return fig2("test_all.json", "fig2_test_leads.png", f"test {p0[:4]}–{p1[:4]} ({n} days per lead); all models fitted on "
                "1982–2015 only; alphas chosen on validation"), fig3()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print(*test_figs(), sep="\n")
    else:
        print(fig2(), figS1(), sep="\n")
