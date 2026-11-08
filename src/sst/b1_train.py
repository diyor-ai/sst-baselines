"""B1: train the LSTM-patch (one region, H=64, one seed), early stopping on VALIDATION loss. Validation only.

The loop is a copy of lstm_train.train_run (frozen file, not edited) with only the data/model input size changed
(tests/test_b1.py checks that with a 1x1 patch and no halo it reproduces lstm_train.train_run exactly).

Run:  python -m sst.b1_train --region kuroshio --seed 0
      python -m sst.b1_train --all --jobs 4
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from sst.b1_data import gather_b1, load_b1, load_halo_sst, make_data_b1
from sst.download import ROOT, load_config
from sst.lstm_model import SSTLSTM
from sst.lstm_train import set_seed

MODELS = ROOT / "models"
RUNS = ROOT / "results" / "b1_runs"
CODE = ["src/sst/b1_data.py", "src/sst/b1_train.py", "src/sst/halo.py", "src/sst/lstm_data.py", "src/sst/lstm_model.py",
        "src/sst/lstm_train.py", "src/sst/climatology.py", "src/sst/download.py", "config.yaml", "config_b1.yaml"]


def val_loss(model, data, chunk=32768):
    """MSE over ALL validation samples and all three leads (standardised units)."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data.val_pairs), chunk):
            x, y = gather_b1(data, data.val_pairs[i:i + chunk])
            err = model(torch.from_numpy(x)) - torch.from_numpy(y)
            sse += float((err.double() ** 2).sum())
            n += err.numel()
    return sse / n


def predict_b1(model, data, flat, chunk=32768):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), chunk):
            x, _ = gather_b1(data, flat[i:i + chunk])
            out.append(model(torch.from_numpy(x)).numpy())
    return np.concatenate(out)


def train_run_b1(data, hidden, seed, lcfg, ckpt_path=None, max_epochs=None, samples_per_epoch=None, verbose=True):
    max_epochs = max_epochs or lcfg["max_epochs"]
    n_draw = samples_per_epoch or lcfg["samples_per_epoch"]
    torch.set_num_threads(lcfg["num_threads"])
    set_seed(seed)
    model = SSTLSTM(hidden, n_in=data.n_in)
    opt = torch.optim.Adam(model.parameters(), lr=lcfg["lr"])
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    bs = lcfg["batch_size"]
    hist = {"train_loss": [], "val_loss": [], "seconds": []}
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        draw = torch.randperm(len(data.train_pairs), generator=gen)[:n_draw].numpy()
        x, y = gather_b1(data, data.train_pairs[draw])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        order = torch.randperm(len(x), generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, len(x), bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
        v = val_loss(model, data)
        hist["train_loss"].append(tot / cnt)
        hist["val_loss"].append(v)
        hist["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  epoch {epoch + 1:2d}  train {tot / cnt:.4f}  val {v:.4f}  {hist['seconds'][-1]:.1f}s", flush=True)
        if v < best - 1e-9:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= lcfg["patience"]:
                break
    res = {"hidden": hidden, "seed": seed, "best_epoch": best_epoch + 1, "best_val_loss": best,
           "epochs_run": len(hist["val_loss"]), "history": hist, "best_state": best_state}
    if ckpt_path is not None:
        torch.save(best_state, ckpt_path)
        res["checkpoint"] = str(ckpt_path.relative_to(ROOT))
        res["sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return res


def weights_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def load_data_b1(region, cfg, b1):
    halo_name = f"{region}_halo"
    sst = load_halo_sst(halo_name, cfg)
    return make_data_b1(sst, cfg, b1["halo"]["width_points"], b1["patch"]["size"])


def run_one(region, seed):
    cfg, b1 = load_config(), load_b1()
    lcfg = cfg["lstm"]
    H = b1["lstm_patch"]["hidden"]
    data = load_data_b1(region, cfg, b1)
    name = f"b1_{region}_patch_H{H}_s{seed}"
    print(f"[{name}] train samples {len(data.train_pairs):,}  val samples {len(data.val_pairs):,}  std {data.std:.3f} C  "
          f"input {data.n_in} channels", flush=True)
    MODELS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prov = {"git_commit": git("rev-parse", "HEAD"), "git_dirty_in_code_files": bool(git("status", "--porcelain", "--", *CODE)),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_sha256": {f: file_sha(f) for f in CODE},
            "data_sha256": {f"data/processed/{n}.nc": file_sha(f"data/processed/{n}.nc") for n in (f"{region}_halo", region)}}
    t0 = time.time()
    res = train_run_b1(data, H, seed, lcfg, ckpt_path=MODELS / f"{name}.pt")
    state = res.pop("best_state")
    res.update({"name": name, "region": region, "model": "lstm_patch", "std_deg_c": data.std, "total_seconds": time.time() - t0,
                "weights_sha256": weights_sha(state),
                "settings": {**{k: lcfg[k] for k in ("seq_len", "samples_per_epoch", "lr", "batch_size", "max_epochs", "patience", "num_threads")},
                             "patch": b1["patch"]["size"], "halo": b1["halo"]["width_points"], "n_in": data.n_in},
                "torch": torch.__version__, "n_train_samples_total": len(data.train_pairs), "n_val_samples": len(data.val_pairs),
                **prov})
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / "b1_train_log.txt", "a") as f:
        f.write(f"{name} best_epoch={res['best_epoch']} epochs_run={res['epochs_run']} best_val_loss={res['best_val_loss']:.5f} "
                f"seconds={res['total_seconds']:.0f} commit={prov['git_commit'][:7]} weights_sha256={res['weights_sha256']}\n")
    print(f"[{name}] best epoch {res['best_epoch']}, val loss {res['best_val_loss']:.4f}, {res['total_seconds']:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    if a.all:
        cfg, b1 = load_config(), load_b1()
        H = b1["lstm_patch"]["hidden"]
        todo = [(r, s) for r in cfg["regions"] for s in b1["lstm_patch"]["seeds"]
                if not (RUNS / f"b1_{r}_patch_H{H}_s{s}.json").exists()]   # finished runs are never overwritten

        def go(job):
            r, s = job
            subprocess.run([sys.executable, "-m", "sst.b1_train", "--region", r, "--seed", str(s)], check=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(go, todo))
    else:
        run_one(a.region, a.seed)


if __name__ == "__main__":
    main()
