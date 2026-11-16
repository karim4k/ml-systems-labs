import csv
import statistics
import time
from pathlib import Path

import mlflow
import psutil
import torch
import yaml
from torch import nn

from common import get_loaders, make_cnn, set_seeds

MODELS = {"cnn": make_cnn}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def build_optimizer(cfg, params):
    o = cfg["optimizer"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"])
    return torch.optim.SGD(params, lr=o["lr"], momentum=o.get("momentum", 0.9))


def build_scheduler(cfg, opt):
    s = cfg.get("scheduler", {"name": "none"})
    n = s["name"]
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=s.get("step_size", 3), gamma=s.get("gamma", 0.5))
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    if n == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=1)
    return None


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        total += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total / n, correct / n


def run(cfg, resume=False, stop_after=None, use_mlflow=True):
    """Train from a single config dict. Saves a checkpoint after every epoch."""
    set_seeds(cfg["seed"])
    train_loader, val_loader, test_loader = get_loaders(
        cfg["batch_size"], cfg["num_workers"], cfg["seed"])
    model = MODELS[cfg["model"]]()
    opt = build_optimizer(cfg, model.parameters())
    sched = build_scheduler(cfg, opt)
    loss_fn = nn.CrossEntropyLoss()
    accum = cfg.get("accum_steps", 1)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{cfg['run_name']}.pt"

    history, start = [], 1
    if resume:
        state = torch.load(ckpt)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if sched is not None and state["sched"] is not None:
            sched.load_state_dict(state["sched"])
        history, start = state["history"], state["epoch"] + 1
        print(f"Resuming {cfg['run_name']} from epoch {start}")

    if use_mlflow:
        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
        mlflow.set_experiment(cfg["experiment"])
        mlflow.start_run(run_name=cfg["run_name"] + ("-resumed" if resume else ""))
        mlflow.log_params(flatten(cfg))
        if resume:
            mlflow.set_tag("resumed_from_epoch", start - 1)

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    interrupted = False
    try:
        for epoch in range(start, cfg["epochs"] + 1):
            train_loader.generator.manual_seed(cfg["seed"] + epoch)  # same order after resume
            model.train()
            t0 = time.perf_counter()
            total, n = 0.0, 0
            nb = len(train_loader)
            opt.zero_grad()
            for i, (x, y) in enumerate(train_loader):
                loss = loss_fn(model(x), y)
                (loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == nb:
                    opt.step()
                    opt.zero_grad()
                total += loss.item() * y.size(0)
                n += y.size(0)
                if i % 20 == 0:
                    peak_rss = max(peak_rss, proc.memory_info().rss)
            train_time = time.perf_counter() - t0

            val_loss, val_acc = evaluate(model, val_loader)
            lr = opt.param_groups[0]["lr"]
            if sched is not None:
                if cfg["scheduler"]["name"] == "plateau":
                    sched.step(val_loss)
                else:
                    sched.step()

            row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                   "val_acc": val_acc, "lr": lr, "train_time_s": train_time,
                   "throughput": n / train_time}
            history.append(row)
            print(f"[{cfg['run_name']}] epoch {epoch} loss={row['train_loss']:.4f} "
                  f"val_acc={val_acc:.4f} lr={lr:.5f} {row['throughput']:.0f} samples/s")
            if use_mlflow:
                for k, v in row.items():
                    if k != "epoch":
                        mlflow.log_metric(k, v, step=epoch)

            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict() if sched is not None else None,
                        "epoch": epoch, "history": history}, ckpt)

            if stop_after is not None and epoch >= stop_after:
                interrupted = True
                print(f"Simulated interruption after epoch {epoch}")
                break

        result = {"run_name": cfg["run_name"], "history": history,
                  "interrupted": interrupted,
                  "peak_rss_mb": peak_rss / 1024 ** 2,
                  "median_throughput": statistics.median(r["throughput"] for r in history)}
        if not interrupted:
            _, test_acc = evaluate(model, test_loader)
            result["test_acc"] = test_acc
            result["final_val_acc"] = history[-1]["val_acc"]
            print(f"[{cfg['run_name']}] test_acc={test_acc:.4f}")

        hist_path = res_dir / f"history_{cfg['run_name']}.csv"
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)
        cfg_path = res_dir / f"config_{cfg['run_name']}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        if use_mlflow:
            mlflow.log_metric("peak_rss_mb", result["peak_rss_mb"])
            mlflow.log_metric("median_throughput", result["median_throughput"])
            if "test_acc" in result:
                mlflow.log_metric("test_acc", result["test_acc"])
            for p in (hist_path, cfg_path, ckpt):
                mlflow.log_artifact(str(p))
        return result
    finally:
        if use_mlflow:
            mlflow.end_run()