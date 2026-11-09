"""Provenance of the B2 runs -> results/b2_manifest<tag>.json: per run commit, dirty flag, settings, sha256 of source/config/data, weights_sha256 (re-computed from the checkpoint)."""
import argparse
import json

import torch

from sst.b2_train import RUNS, file_sha, weights_sha
from sst.download import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    out = {"models": a.models.split(","), "runs": {}}
    for m in out["models"]:
        for f in sorted(RUNS.glob(f"b2_{m}_*.json")):
            j = json.loads(f.read_text())
            ck = ROOT / j["checkpoint"]
            assert file_sha(j["checkpoint"]) == j["sha256"], f.name
            assert weights_sha(torch.load(ck, map_location="cpu")) == j["weights_sha256"], f.name
            out["runs"][j["name"]] = {k: j[k] for k in ("settings", "torch", "best_epoch", "epochs_run", "best_val_loss", "n_train_samples", "n_val_samples", "n_params",
                                                        "git_commit", "git_dirty_in_code_files", "started_utc", "code_sha256", "data_sha256", "weights_sha256")}
            out["runs"][j["name"]]["checkpoint_file_sha256"] = j["sha256"]
    (ROOT / "results" / f"b2_manifest{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote results/b2_manifest{a.tag}.json:", len(out["runs"]), "runs; dirty flags:", sorted({v["git_dirty_in_code_files"] for v in out["runs"].values()}),
          "commits:", sorted({v["git_commit"][:7] for v in out["runs"].values()}))


if __name__ == "__main__":
    main()
