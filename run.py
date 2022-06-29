import argparse
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from dpl.processor.engine import Engine
from dpl.processor.nodes.base import NodeExecReport

DEFAULT_OUTPUT_DIR = "/data/datasets/dpl_cache"
DEFAULT_REPORT_NAME = "report.json"


def abs_path(path: str) -> Path:
    return Path(path).expanduser().absolute()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=abs_path, required=True)
    parser.add_argument("--inputs", type=abs_path, required=True)
    parser.add_argument("--output_dir", type=abs_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_name", type=str, default=DEFAULT_REPORT_NAME)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--test_run", action="store_true")
    return parser.parse_args()


def read_inputs(path) -> Dict[str, Path]:
    with open(path) as ifile:
        inputs = json.load(ifile)
    return {key: Path(path) for key, path in inputs.items()}


def save_report(path: Path, reports: List[NodeExecReport]) -> None:
    final_report = [report.to_dict() for report in reports]
    with open(path, "w") as ofile:
        json.dump(final_report, ofile, indent=2)


def main():
    args = parse_args()
    engine = Engine(args.graph, args.output_dir)
    inputs = read_inputs(args.inputs)
    engine.init(inputs)
    reports = engine.execute(args.verbose, args.chunk_size, args.test_run)
    save_report(args.output_dir / args.report_name, reports)


if __name__ == "__main__":
    main()
