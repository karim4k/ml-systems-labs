import argparse
import yaml
from pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lab06/config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()