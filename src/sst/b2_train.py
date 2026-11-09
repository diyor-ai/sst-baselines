"""B2: train one whole-region model (unet | convlstm) for one region, channel width C and seed; early stopping on VALIDATION loss.

Run:  python -m sst.b2_train --model unet --region kuroshio --C 16 --seed 0
      python -m sst.b2_train --trial --region kuroshio        # 1 epoch for every model/C (results/b2_trials/), nothing else written
      python -m sst.b2_train --all --jobs 4                   # the 24 runs (2 models x 2 regions x C{16,32} x seeds 0,1,2); finished runs are skipped
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
import yaml
from torch import nn

from sst.b2_data import crop, gather_x, gather_y, load_data_b2
from sst.b2_models import build
from sst.download import ROOT, load_config
from sst.lstm_train import set_seed

MODELS = ROOT / "models"
RUNS = ROOT / "results" / "b2_runs"
TRIALS = ROOT / "results" / "b2_trials"
CODE = ["src/sst/b2_data.py", "src/sst/b2_models.py", "src/sst/b2_train.py", "src/sst/b1_data.py", "src/sst/lstm_data.py",
        "src/sst/lstm_train.py", "src/sst/climatology.py", "src/sst/download.py", "src/sst/halo.py", "config.yaml", "config_b1.yaml", "config_b2.yaml"]


def load_b2(path=None):
    with open(path or ROOT / "config_b2.yaml") as f:
        return yaml.safe_load(f)


def predict_b2(model, data, issue, chunk=64):
    """(n, 3, 40, 40) standardised anomaly predictions (cropped) for issue indices `issue`."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(issue), chunk):
            out.append(crop(model(torch.from_numpy(gather_x(data, issue[i:i + chunk])))).numpy())
    return np.concatenate(out)


def val_loss(model, data, chunk=64):
    """MSE over all validation issue dates, the inner 40x40 and the 3 leads (standardised units)."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data.val_issue), chunk):
            t = data.val_issue[i:i + chunk]
            err = crop(model(torch.from_numpy(gather_x(data, t)))) - torch.from_numpy(gather_y(data, t))
            sse += float((err.double() ** 2).sum())
            n += err.numel()
    return sse / n


def train_run_b2(data, model_name, C, seed, tcfg, ckpt_path=None, max_epochs=None, verbose=True, max_train=None):
    max_epochs = max_epochs or tcfg["max_epochs"]
    torch.set_num_threads(tcfg["num_threads"])
    set_seed(seed)
    model = build(model_name, C)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    bs = tcfg["batch_size"]
    train = data.train_issue if max_train is None else data.train_issue[:max_train]
    hist = {"train_loss": [], "val_loss": [], "seconds": []}
    best, best_epoch, best_state, bad = float("inf"), -1, None, 0
    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        order = train[torch.randperm(len(train), generator=gen).numpy()]
        tot, cnt = 0.0, 0
        for i in range(0, len(order), bs):
            t = order[i:i + bs]
            x, y = torch.from_numpy(gather_x(data, t)), torch.from_numpy(gather_y(data, t))
            opt.zero_grad()
            loss = loss_fn(crop(model(x)), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(t)
            cnt += len(t)
        v = val_loss(model, data)
        hist["train_loss"].append(tot / cnt)
        hist["val_loss"].append(v)
        hist["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  epoch {epoch + 1:2d}  train {tot / cnt:.4f}  val {v:.4f}  {hist['seconds'][-1]:.1f}s", flush=True)
        if v < best - 1e-9:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    res = {"model": model_name, "C": C, "seed": seed, "best_epoch": best_epoch + 1, "best_val_loss": best,
           "epochs_run": len(hist["val_loss"]), "history": hist, "best_state": best_state}
    if ckpt_path is not None:
        torch.save(best_state, ckpt_path)
        res["checkpoint"] = str(ckpt_path.relative_to(ROOT))
        res["sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return res


def time_trial(data, model_name, C, seed, tcfg, n_batches=20, n_val=None):
    """EXTRAPOLATED epoch time (used only when a full 1-epoch trial would itself take hours): real training batches of the real
    sampler (n_batches after 2 warm-up batches) + a real validation pass over `n_val` issue dates, scaled to the full epoch / full validation set."""
    torch.set_num_threads(tcfg["num_threads"])
    set_seed(seed)
    model = build(model_name, C)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    gen = torch.Generator().manual_seed(seed)
    bs = tcfg["batch_size"]
    order = data.train_issue[torch.randperm(len(data.train_issue), generator=gen).numpy()]
    model.train()
    secs = []
    for i in range(n_batches + 2):
        t = order[i * bs:(i + 1) * bs]
        t0 = time.time()
        x, y = torch.from_numpy(gather_x(data, t)), torch.from_numpy(gather_y(data, t))
        opt.zero_grad()
        nn.MSELoss()(crop(model(x)), y).backward()
        opt.step()
        secs.append(time.time() - t0)
    per_batch = float(np.mean(secs[2:]))
    n_val = n_val or len(data.val_issue)
    vd = type(data)(**{**data.__dict__, "val_issue": data.val_issue[:n_val]})
    t0 = time.time()
    val_loss(model, vd)
    val_s = (time.time() - t0) * len(data.val_issue) / n_val
    n_b = int(np.ceil(len(data.train_issue) / bs))
    return {"method": "EXTRAPOLATED from %d timed training batches + validation on %d dates" % (n_batches, n_val), "sec_per_batch": per_batch,
            "batch_seconds": secs, "n_batches_per_epoch": n_b, "val_seconds_full": val_s, "epoch_seconds_est": per_batch * n_b + val_s}


def weights_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def file_sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def git(*x):
    return subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance(region):
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty_in_code_files": bool(git("status", "--porcelain", "--", *CODE)),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_sha256": {f: file_sha(f) for f in CODE},
            "data_sha256": {f"data/processed/{n}.nc": file_sha(f"data/processed/{n}.nc") for n in (f"{region}_halo", region)}}


def run_name(model, region, C, seed):
    return f"b2_{model}_{region}_C{C}_s{seed}"


def run_one(model, region, C, seed, trial=False, timed=False):
    cfg, b2 = load_config(), load_b2()
    tcfg = b2["train"]
    data = load_data_b2(region, cfg)
    name = run_name(model, region, C, seed)
    print(f"[{name}] train samples {len(data.train_issue):,}  val samples {len(data.val_issue):,}  std {data.std:.3f} C", flush=True)
    prov = provenance(region)
    t0 = time.time()
    if trial:
        TRIALS.mkdir(parents=True, exist_ok=True)
        if timed:
            res = time_trial(data, model, C, seed, tcfg)
            res.update({"name": name, "region": region, "trial": "timed (extrapolated)", "n_train": len(data.train_issue), "n_val": len(data.val_issue),
                        "n_params": sum(p.numel() for p in build(model, C).parameters()), "torch": torch.__version__, "num_threads": tcfg["num_threads"], **prov})
            (TRIALS / f"{name}_timed.json").write_text(json.dumps(res, indent=1))
            print(f"[{name}] timed trial: {res['sec_per_batch']:.2f} s/batch -> est. {res['epoch_seconds_est']:.0f} s/epoch", flush=True)
            return
        res = train_run_b2(data, model, C, seed, tcfg, max_epochs=1)
        res.pop("best_state")
        res.update({"name": name, "region": region, "trial": True, "total_seconds": time.time() - t0, "n_train": len(data.train_issue),
                    "n_val": len(data.val_issue), "n_params": sum(p.numel() for p in build(model, C).parameters()), "torch": torch.__version__,
                    "num_threads": tcfg["num_threads"], "concurrent_jobs_note": "see results/b2_trial_summary.md", **prov})
        (TRIALS / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"[{name}] trial: {res['history']['seconds'][0]:.0f} s/epoch (incl. validation)", flush=True)
        return
    MODELS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    res = train_run_b2(data, model, C, seed, tcfg, ckpt_path=MODELS / f"{name}.pt")
    state = res.pop("best_state")
    res.update({"name": name, "region": region, "std_deg_c": data.std, "total_seconds": time.time() - t0, "weights_sha256": weights_sha(state),
                "settings": {**tcfg, "C": C, "model": model}, "torch": torch.__version__, "n_train_samples": len(data.train_issue),
                "n_val_samples": len(data.val_issue), "n_params": sum(p.numel() for p in build(model, C).parameters()), **prov})
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results" / "b2_train_log.txt", "a") as f:
        f.write(f"{name} best_epoch={res['best_epoch']} epochs_run={res['epochs_run']} best_val_loss={res['best_val_loss']:.5f} "
                f"seconds={res['total_seconds']:.0f} commit={prov['git_commit'][:7]} weights_sha256={res['weights_sha256']}\n")
    print(f"[{name}] best epoch {res['best_epoch']}, val loss {res['best_val_loss']:.4f}, {res['total_seconds']:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--region")
    p.add_argument("--C", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", action="store_true")
    p.add_argument("--only", help="with --trial: run only this model")
    p.add_argument("--timed", action="store_true", help="with --trial: extrapolate the epoch time from a few real batches instead of a full epoch")
    p.add_argument("--all", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    cfg, b2 = load_config(), load_b2()
    if a.trial and not a.model:
        jobs = [(m, a.region, C, 0) for m in ("unet", "convlstm") if a.only in (None, m) for C in b2["C_grid"]]
    elif a.all:
        jobs = [(m, r, C, s) for m in ("unet", "convlstm") for r in cfg["regions"] for C in b2["C_grid"] for s in b2["seeds"]
                if not (RUNS / f"{run_name(m, r, C, s)}.json").exists()]          # finished runs are never overwritten
    else:
        jobs = [(a.model, a.region, a.C, a.seed)]
    if len(jobs) == 1:
        run_one(*jobs[0], trial=a.trial, timed=a.timed)
        return

    def go(job):
        m, r, C, s = job
        cmd = [sys.executable, "-m", "sst.b2_train", "--model", m, "--region", r, "--C", str(C), "--seed", str(s)] + (["--trial"] if a.trial else []) + (["--timed"] if a.timed else [])
        subprocess.run(cmd, check=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, jobs))


if __name__ == "__main__":
    main()
