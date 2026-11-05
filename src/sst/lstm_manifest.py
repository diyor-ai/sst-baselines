"""Provenance of the LSTM runs -> results/lstm_manifest.json.

Per run: settings, torch version, checkpoint file sha256 AND a weights hash that does not depend on zip metadata
(sha256 over the tensors in sorted key order: key name, dtype, shape, raw bytes). Plus the git commit and the sha256 of
every source/config/data file the training used. `--expect FILE` checks the code hashes against a `sha256sum` listing
taken while training was running (so we can prove the code did not change between runs)."""
import argparse
import hashlib
import json
import subprocess

import torch

from sst.download import ROOT, load_config

CODE = ["src/sst/lstm_data.py", "src/sst/lstm_model.py", "src/sst/lstm_train.py", "src/sst/climatology.py",
        "src/sst/download.py", "config.yaml", "data/processed/kuroshio.nc", "data/processed/subtropical.nc"]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_sha(path):
    sd = torch.load(path, map_location="cpu")
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expect", help="sha256sum listing taken during training")
    a = p.parse_args()
    cfg = load_config()
    code = {f: file_sha(ROOT / f) for f in CODE}
    if a.expect:
        exp = {l.split()[1]: l.split()[0] for l in open(a.expect) if l.strip()}
        assert exp == code, {f: (exp.get(f), code[f]) for f in code if exp.get(f) != code[f]}
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    runs = {}
    for r in cfg["regions"]:
        for H in cfg["lstm"]["hidden_sizes"]:
            for s in cfg["lstm"]["seeds"]:
                name = f"{r}_H{H}_s{s}"
                j = json.loads((ROOT / "results" / "lstm_runs" / f"{name}.json").read_text())
                ck = ROOT / j["checkpoint"]
                assert file_sha(ck) == j["sha256"], name
                runs[name] = {"settings": j["settings"], "torch": j["torch"], "hidden": H, "seed": s,
                              "best_epoch": j["best_epoch"], "epochs_run": j["epochs_run"],
                              "n_train_samples_total": j["n_train_samples_total"], "n_val_samples": j["n_val_samples"],
                              "checkpoint_file_sha256": j["sha256"], "weights_sha256": weights_sha(ck)}
    out = {"git_commit": git("rev-parse", "HEAD"), "git_dirty_in_code_files": bool(git("status", "--porcelain", "--", *CODE[:6])),
           "code_sha256": code, "code_checked_against_training_time_listing": bool(a.expect),
           "config_lstm": cfg["lstm"], "runs": runs}
    (ROOT / "results" / "lstm_manifest.json").write_text(json.dumps(out, indent=1))
    print("wrote results/lstm_manifest.json,", len(runs), "runs")


if __name__ == "__main__":
    main()
