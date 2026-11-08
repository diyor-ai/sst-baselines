"""Provenance of the B1 runs -> results/b1_manifest.json (git commit, settings, sha256 of source/config/data files per run,
weights_sha256 for the LSTM-patch runs, sha256 of the stored predictions for the Ridge runs)."""
import hashlib
import json
import subprocess

import torch

from sst.b1_train import CODE, RUNS, file_sha, weights_sha
from sst.download import ROOT, load_config
from sst.b1_data import load_b1


def main():
    cfg, b1 = load_config(), load_b1()
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    code = CODE + ["src/sst/b1_ridge.py", "src/sst/b1_eval.py", "src/sst/ridge.py", "src/sst/metrics.py"]
    out = {"git_commit_at_manifest": git("rev-parse", "HEAD"), "code_sha256_at_manifest": {f: file_sha(f) for f in code},
           "data_sha256": {f"data/processed/{n}.nc": file_sha(f"data/processed/{n}.nc")
                           for r in cfg["regions"] for n in (r, f"{r}_halo")},
           "config_b1": b1, "ridge_runs": {}, "lstm_patch_runs": {}}
    rv = json.loads((ROOT / "results" / "b1_ridge_val.json").read_text())
    for r in cfg["regions"]:
        npz = f"data/derived/b1_ridge_{r}.npz"
        out["ridge_runs"][r] = {"git_commit_at_run": rv["git_commit"], "git_dirty_at_run": rv["git_dirty"], "predictions_file": npz,
                                "predictions_file_sha256": file_sha(npz),
                                "models": {m: {h: {k: v[h][k] for k in ("alpha", "alpha_at_grid_edge", "n_features", "n_train_rows")}
                                               for h in v} for m, v in rv["regions"][r].items()}}
    for r in cfg["regions"]:
        for s in b1["lstm_patch"]["seeds"]:
            name = f"b1_{r}_patch_H{b1['lstm_patch']['hidden']}_s{s}"
            j = json.loads((RUNS / f"{name}.json").read_text())
            ck = ROOT / j["checkpoint"]
            assert file_sha(j["checkpoint"]) == j["sha256"], name
            sd = torch.load(ck, map_location="cpu")
            assert weights_sha(sd) == j["weights_sha256"], name
            out["lstm_patch_runs"][name] = {k: j[k] for k in ("settings", "torch", "best_epoch", "epochs_run", "best_val_loss", "n_train_samples_total",
                                                              "n_val_samples", "git_commit", "git_dirty_in_code_files", "started_utc",
                                                              "code_sha256", "data_sha256", "sha256", "weights_sha256")}
            out["lstm_patch_runs"][name]["checkpoint_file_sha256"] = out["lstm_patch_runs"][name].pop("sha256")
    (ROOT / "results" / "b1_manifest.json").write_text(json.dumps(out, indent=1))
    print("wrote results/b1_manifest.json:", len(out["lstm_patch_runs"]), "LSTM runs,", len(out["ridge_runs"]), "ridge regions")
    print("dirty flags:", {n: v["git_dirty_in_code_files"] for n, v in out["lstm_patch_runs"].items()}, {r: v["git_dirty_at_run"] for r, v in out["ridge_runs"].items()})


if __name__ == "__main__":
    main()
