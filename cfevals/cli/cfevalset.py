from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfevals.cli.cfeval import run_eval
from cfevals.registry import Registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an eval set")
    parser.add_argument("eval_set_id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    registry = Registry().load()
    eval_set = registry.get_eval_set(args.eval_set_id)
    eval_ids = eval_set["evals"]

    for eval_id in eval_ids:
        if args.resume:
            output_dir = Path("outputs") / eval_id
            if output_dir.exists():
                continue
        run_eval(eval_id, None, None)


if __name__ == "__main__":
    main()
