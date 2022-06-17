import argparse
from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from dpl.processor.engine import Engine
from dpl.processor.nodes.base import NodeExecReport

DEFAULT_OUTPUT_DIR = "/data/datasets/dpl_cache"
DEFAULT_REPORT_PATH = f"{DEFAULT_OUTPUT_DIR}/report.json"


def abs_path(path: str) -> Path:
    return Path(path).expanduser().absolute()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=abs_path, required=True)
    parser.add_argument("--inputs", type=abs_path, required=True)
    parser.add_argument("--output_dir", type=abs_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_path", type=abs_path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--test_run", type=bool, default=True)
    return parser.parse_args()


def read_inputs(path) -> Dict[str, Path]:
    with open(path) as ifile:
        inputs = json.load(ifile)
    return {key: Path(path) for key, path in inputs.items()}


def save_report(path: Path, reports: List[NodeExecReport]) -> None:
    def _to_str_list(items: Optional[List[Any]]) -> List[str]:
        if items is None:
            return []
        return list(map(str, items))

    final_report = [
        {
            "name": report.name,
            "start": report.start,
            "total": report.total,
            "missing_inputs": _to_str_list(report.missing_inputs),
            "error_inputs": _to_str_list(report.error_inputs),
            "error_messages": _to_str_list(report.error_messages),
        }
        for report in reports
    ]
    with open(path, "w") as ofile:
        json.dump(final_report, ofile, indent=2)


def main():
    args = parse_args()
    engine = Engine(args.graph, args.output_dir)
    inputs = read_inputs(args.inputs)
    engine.init(inputs)
    reports = engine.execute(args.verbose, args.chunk_size, args.test_run)
    save_report(args.report_path, reports)


if __name__ == "__main__":
    main()