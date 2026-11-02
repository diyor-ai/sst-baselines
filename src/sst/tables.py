"""Print short tables from results JSON (no computation of new scores)."""
import json

from sst.download import ROOT

R = ROOT / "results"


def main():
    d = json.loads((R / "val_all.json").read_text())
    b = json.loads((R / "val_bias.json").read_text())
    print("VALIDATION 2016-2019: RMSE (degC) | skill vs anomaly persistence")
    for reg, rd in d["regions"].items():
        print(f"\n{reg}")
        print(f"{'method':22s}" + "".join(f"  h={h:<2s} RMSE  skill" for h in rd))
        for m in d["methods"]:
            print(f"{m:22s}" + "".join(
                f"  {rd[h]['methods'][m]['overall']['rmse']:10.3f} {rd[h]['skill'][m]['rmse_skill_vs_anomaly_persistence']:+7.3f}"
                for h in rd))
    print("\nBIAS (forecast - observed, degC) | bias^2 share of MSE (%) ; lead 1/7/30")
    for reg, rd in b["bias_by_region"].items():
        print(f"\n{reg}")
        for m in d["methods"]:
            print(f"{m:22s}" + "".join(f"  {rd[h][m]['bias']:+7.3f} {rd[h][m]['bias_sq_pct_of_mse']:6.1f}%" for h in rd))
    for reg, a in b["area_mean_anomaly"].items():
        print(reg, "mean anomaly train %.3f val %.3f" % (a["mean_train"], a["mean_val"]),
              {y: round(v, 2) for y, v in a["yearly_mean"].items() if int(y) >= 2013})
    rs = json.loads((R / "region_stats.json").read_text())
    print({k: v["point_anomaly_std_cosw_mean"] for k, v in rs.items() if isinstance(v, dict) and "point_anomaly_std_cosw_mean" in v})
    a = json.loads((R / "ridge_alphas.json").read_text())
    print({r: {h: i["alpha"] for h, i in v.items()} for r, v in a["regions"].items()})
    for reg, rd in d["regions"].items():
        print(reg, "ACC temporal/spatial (h=1,7,30) ridge:", [(round(rd[h]['methods']['ridge']['overall']['acc'], 3),
              round(rd[h]['methods']['ridge']['overall']['acc_spatial'], 3)) for h in rd])


if __name__ == "__main__":
    main()
