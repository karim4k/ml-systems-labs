import csv
import sys
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from pipeline import run

R = "lab06/results"


def base():
    with open("lab06/config.yaml") as f:
        return yaml.safe_load(f)


def _job(cfg, kw):
    return run(cfg, **kw)


def execute(cfg, **kw):
    with mp.get_context("spawn").Pool(1) as p:
        return p.apply(_job, (cfg, kw))


def save(rows, name):
    with open(f"{R}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


def batch_study():
    rows = []
    for bs in [32, 64, 128, 256, 512]:
        cfg = base()
        cfg.update(run_name=f"batch{bs}", batch_size=bs, epochs=5)
        res = execute(cfg)
        rows.append({"batch_size": bs,
                     "median_throughput_samples_s": round(res["median_throughput"]),
                     "peak_rss_mb": round(res["peak_rss_mb"], 1),
                     "final_val_acc": round(res["final_val_acc"], 4),
                     "test_acc": round(res["test_acc"], 4)})
    save(rows, "batch_study")


def sched_study():
    setups = [("adam_none", "adam", "none"), ("adam_step", "adam", "step"),
              ("adam_cosine", "adam", "cosine"), ("adam_plateau", "adam", "plateau"),
              ("sgd_none", "sgd", "none"), ("sgd_cosine", "sgd", "cosine")]
    rows, curves = [], {}
    for name, opt, sch in setups:
        cfg = base()
        cfg.update(run_name=name, epochs=10)
        cfg["optimizer"] = {"name": opt, "lr": 0.001 if opt == "adam" else 0.01,
                            "momentum": 0.9}
        cfg["scheduler"] = {"name": sch, "step_size": 3, "gamma": 0.5}
        res = execute(cfg)
        curves[name] = [h["val_acc"] for h in res["history"]]
        rows.append({"run": name, "optimizer": opt, "scheduler": sch,
                     "final_train_loss": round(res["history"][-1]["train_loss"], 4),
                     "final_val_acc": round(res["final_val_acc"], 4),
                     "test_acc": round(res["test_acc"], 4)})
    save(rows, "scheduler_optimizer_study")
    plt.figure(figsize=(7, 4))
    for n, v in curves.items():
        plt.plot(range(1, len(v) + 1), v, label=n)
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{R}/scheduler_curves.png", dpi=150)


def accum_study():
    rows, curves = [], {}
    for name, bs, acc in [("batch256", 256, 1), ("batch64_accum4", 64, 4),
                          ("batch64_noaccum", 64, 1)]:
        cfg = base()
        cfg.update(run_name=name, batch_size=bs, accum_steps=acc, epochs=5)
        res = execute(cfg)
        curves[name] = [h["train_loss"] for h in res["history"]]
        rows.append({"run": name, "micro_batch": bs, "accum_steps": acc,
                     "final_train_loss": round(res["history"][-1]["train_loss"], 4),
                     "final_val_acc": round(res["final_val_acc"], 4),
                     "test_acc": round(res["test_acc"], 4)})
    save(rows, "accumulation_study")
    plt.figure(figsize=(7, 4))
    for n, v in curves.items():
        plt.plot(range(1, len(v) + 1), v, marker="o", label=n)
    plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{R}/accumulation_curves.png", dpi=150)


def resume_test():
    cfg = base()
    cfg.update(epochs=6)
    full = dict(cfg, run_name="resume_full")
    res_full = execute(full)
    part = dict(cfg, run_name="resume_test")
    execute(part, stop_after=3)             # interrupted after epoch 3
    res_res = execute(part, resume=True)    # continues from the checkpoint
    a = [h["train_loss"] for h in res_full["history"]]
    b = [h["train_loss"] for h in res_res["history"]]
    diffs = [abs(x - y) for x, y in zip(a, b)]
    rows = [{"epoch": i + 1, "uninterrupted_loss": round(a[i], 6),
             "resumed_loss": round(b[i], 6), "abs_diff": f"{diffs[i]:.2e}"}
            for i in range(len(a))]
    save(rows, "resume_test")
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 7), a, label="uninterrupted")
    plt.plot(range(1, 7), b, "--o", label="interrupted at 3, resumed")
    plt.axvline(3, color="gray", ls=":")
    plt.xlabel("Epoch"); plt.ylabel("Train loss"); plt.legend()
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{R}/resume_curves.png", dpi=150)


if __name__ == "__main__":
    {"batch": batch_study, "sched": sched_study,
     "accum": accum_study, "resume": resume_test}[sys.argv[1]]()