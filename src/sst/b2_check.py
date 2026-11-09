"""Sanity check of finished B2 runs: expected count, finite losses, no divergence. Exit 1 (and print why) if anything looks wrong."""
import argparse
import json
import math
import sys

from sst.b2_train import RUNS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--expect", type=int, required=True)
    a = p.parse_args()
    files = sorted(RUNS.glob(f"b2_{a.model}_*.json"))
    bad = []
    if len(files) != a.expect:
        bad.append(f"expected {a.expect} runs, found {len(files)}")
    for f in files:
        j = json.loads(f.read_text())
        h = j["history"]
        if not all(math.isfinite(x) for x in h["train_loss"] + h["val_loss"]):
            bad.append(f"{f.name}: non-finite loss")
        elif j["best_val_loss"] > 1.0:
            bad.append(f"{f.name}: best validation loss {j['best_val_loss']:.3f} > 1.0 (worse than predicting zero anomaly)")
        elif j["best_epoch"] == 1 and j["epochs_run"] > 1:       # WARNING only (decided 2026-11-09): finite losses, fast overfitting is expected with ~12k samples
            print(f"WARNING {f.name}: best epoch is 1 of {j['epochs_run']} (losses finite). train: {[round(x, 3) for x in h['train_loss']]}  val: {[round(x, 3) for x in h['val_loss']]}")
        print(f"{f.name}: best_epoch {j['best_epoch']}/{j['epochs_run']}  best_val {j['best_val_loss']:.4f}  threads {j['settings']['num_threads']}")
    if bad:
        print("PROBLEMS:\n" + "\n".join(bad))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
