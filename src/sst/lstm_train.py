"""Train one LSTM (one region, one hidden size, one seed) with early stopping on VALIDATION loss.

Run:  python -m sst.lstm_train --region kuroshio --hidden 32 --seed 0
      python -m sst.lstm_train --trial --region kuroshio        # 1 epoch, prints seconds/epoch
      python -m sst.lstm_train --all --jobs 4                   # the 12 runs (2 regions x H{32,64} x seeds 0,1,2)
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import xarray as xr
from torch import nn

from sst.download import ROOT, load_config
from sst.lstm_data import gather, make_data
from sst.lstm_model import SSTLSTM

MODELS = ROOT / "models"
RUNS = ROOT / "results" / "lstm_runs"


def set_seed(seed):
    """Make Python, numpy and torch random numbers repeatable, and ask torch for deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict(model, data, flat, chunk=32768):
    """Predictions (n, 3) in standardised units for the samples `flat` (no gradients, evaluation mode)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(flat), chunk):
            x, _ = gather(data, flat[i:i + chunk])
            out.append(model(torch.from_numpy(x)).numpy())
    return np.concatenate(out)


def val_loss(model, data, chunk=32768):
    """MSE over ALL validation samples and all three leads (standardised units)."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data.val_pairs), chunk):
            x, y = gather(data, data.val_pairs[i:i + chunk])
            err = model(torch.from_numpy(x)) - torch.from_numpy(y)
            sse += float((err.double() ** 2).sum())
            n += err.numel()
    return sse / n


def train_run(data, hidden, seed, lcfg, ckpt_path=None, max_epochs=None, samples_per_epoch=None, verbose=True):
    """Train and return a result dict (history, best epoch, timings). The best-validation weights are kept in
    memory (`result['best_state']`) and, if `ckpt_path` is given, written to disk."""
    max_epochs = max_epochs or lcfg["max_epochs"]
    n_draw = samples_per_epoch or lcfg["samples_per_epoch"]
    torch.set_num_threads(lcfg["num_threads"])
    set_seed(seed)                                    # 1. seeds first, so the weight initialisation is repeatable
    model = SSTLSTM(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lcfg["lr"])
    loss_fn = nn.MSELoss()                            # mean over batch AND the 3 leads
    gen = torch.Generator().manual_seed(seed)         # separate generator for the random draw of training samples
    bs = lcfg["batch_size"]
    hist = {"train_loss": [], "val_loss": [], "seconds": []}
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()                                 # 2. training mode
        draw = torch.randperm(len(data.train_pairs), generator=gen)[:n_draw].numpy()   # fixed-size random subset
        x, y = gather(data, data.train_pairs[draw])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        order = torch.randperm(len(x), generator=gen)                                   # shuffle inside the epoch
        tot, cnt = 0.0, 0
        for i in range(0, len(x), bs):                # 3. one pass = many mini-batches
            idx = order[i:i + bs]
            opt.zero_grad()                           #    forget gradients of the previous batch
            loss = loss_fn(model(x[idx]), y[idx])     #    forward pass + loss
            loss.backward()                           #    backward pass: gradient of loss w.r.t. every weight
            opt.step()                                #    Adam update of the weights
            tot += loss.item() * len(idx)
            cnt += len(idx)
        v = val_loss(model, data)                     # 4. score on ALL validation samples (no learning here)
        hist["train_loss"].append(tot / cnt)
        hist["val_loss"].append(v)
        hist["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  epoch {epoch + 1:2d}  train {tot / cnt:.4f}  val {v:.4f}  {hist['seconds'][-1]:.1f}s", flush=True)
        if v < best - 1e-9:                           # 5. early stopping: keep the best weights, count bad epochs
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


def load_sst(region, cfg):
    """Region SST cut at the validation end BEFORE anything else (the test period is never loaded)."""
    v1 = cfg["split"]["val"][1]
    sst = xr.open_dataset(ROOT / "data" / "processed" / f"{region}.nc")["sst"].sel(time=slice(None, v1)).load()
    assert str(sst.time.max().values)[:10] <= v1
    return sst


def load_data(region, cfg):
    return make_data(load_sst(region, cfg), cfg)


def run_one(region, hidden, seed, trial=False):
    cfg = load_config()
    lcfg = cfg["lstm"]
    data = load_data(region, cfg)
    print(f"[{region} H={hidden} seed={seed}] train samples {len(data.train_pairs):,}  val samples {len(data.val_pairs):,}  "
          f"std {data.std:.3f} C", flush=True)
    MODELS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"{region}_H{hidden}_s{seed}"
    t0 = time.time()
    if trial:
        res = train_run(data, hidden, seed, lcfg, max_epochs=1)
        out = {"region": region, "hidden": hidden, "seed": seed, "seconds_per_epoch": res["history"]["seconds"][0],
               "n_train_samples_total": len(data.train_pairs), "n_val_samples": len(data.val_pairs)}
        (ROOT / "results" / f"lstm_trial_{region}_H{hidden}.json").write_text(json.dumps(out, indent=1))
        print(out)
        return
    res = train_run(data, hidden, seed, lcfg, ckpt_path=MODELS / f"{name}.pt")
    res.pop("best_state")
    res.update({"region": region, "std_deg_c": data.std, "total_seconds": time.time() - t0,
                "settings": {k: lcfg[k] for k in ("seq_len", "samples_per_epoch", "lr", "batch_size", "max_epochs", "patience", "num_threads")},
                "torch": torch.__version__, "n_train_samples_total": len(data.train_pairs), "n_val_samples": len(data.val_pairs)})
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / "lstm_train_log.txt", "a") as f:
        f.write(f"{name} best_epoch={res['best_epoch']} epochs_run={res['epochs_run']} best_val_loss={res['best_val_loss']:.5f} "
                f"seconds={res['total_seconds']:.0f} sha256={res['sha256']}\n")
    print(f"[{name}] best epoch {res['best_epoch']}, val loss {res['best_val_loss']:.4f}, {res['total_seconds']:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region")
    p.add_argument("--hidden", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    if a.all:
        cfg = load_config()
        l = cfg["lstm"]
        todo = [(r, h, s) for r in cfg["regions"] for h in l["hidden_sizes"] for s in l["seeds"]
                if not (RUNS / f"{r}_H{h}_s{s}.json").exists()]   # finished runs are never overwritten

        def go(job):
            r, h, s = job
            subprocess.run([sys.executable, "-m", "sst.lstm_train", "--region", r, "--hidden", str(h), "--seed", str(s)], check=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(go, todo))
    else:
        run_one(a.region, a.hidden, a.seed, a.trial)


if __name__ == "__main__":
    main()
