"""Quick profile of one training step (batch 32) of a B2 model: data gather / forward / backward / optimiser, with deterministic algorithms on/off
and a given thread count. Reads training data only (days <= 2015-12-31 are what the sampler uses; validation data is loaded but not touched).
Run:  python -m sst.b2_profile --model convlstm_ed --C 16 --threads 4 --det 1 --batches 6      -> prints one JSON line (timing is not a result)
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from sst.b2_data import crop, gather_x, gather_y, load_data_b2
from sst.b2_models import build
from sst.b2_train import load_b2
from sst.download import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="convlstm_ed")
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--region", default="kuroshio")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--det", type=int, default=1)
    p.add_argument("--batches", type=int, default=6)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(bool(a.det))
    torch.manual_seed(0)
    data = load_data_b2(a.region, load_config())
    tc = load_b2()["train"]
    model = build(a.model, a.C)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    order = data.train_issue[np.random.default_rng(0).permutation(len(data.train_issue))]
    T = {"gather": [], "forward": [], "backward": [], "optimiser": []}
    for i in range(a.batches + 2):
        t = order[i * tc["batch_size"]:(i + 1) * tc["batch_size"]]
        t0 = time.perf_counter()
        x, y = torch.from_numpy(gather_x(data, t)), torch.from_numpy(gather_y(data, t))
        t1 = time.perf_counter()
        opt.zero_grad()
        loss = nn.functional.mse_loss(crop(model(x)), y)
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        opt.step()
        t4 = time.perf_counter()
        if i >= 2:                                           # 2 warm-up batches
            for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                T[k].append(v)
    m = {k: float(np.mean(v)) for k, v in T.items()}
    m["total"] = sum(m.values())
    print(json.dumps({"model": a.model, "C": a.C, "threads": a.threads, "deterministic": bool(a.det), "sec_per_batch": m, "last_loss": float(loss)}))


if __name__ == "__main__":
    main()
