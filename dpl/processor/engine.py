import collections
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dpl.processor.nodes import get_node_classes
from dpl.processor.nodes.base import BaseNode, NodeExecReport
from dpl.processor.datatype import DataType
from dpl import common


class Engine:
    def __init__(
        self, nodes: Union[Path, List[Dict[str, Any]]], output_dir: Path
    ) -> None:
        self._set_nodes(nodes)
        self.output_dir = output_dir

    def init(self, inputs: Dict[str, Path]) -> None:
        dts = {dt.key: dt for dt in DataType}
        paths = {
            key: common.listdir(
                path,
                ext=None if dts[key].is_sequential() else dts[key].extensions(),
                recursive=False if dts[key].is_sequential() else True,
            )
            for key, path in inputs.items()
        }
        input_names = self._deduce_names(inputs, paths)
        name_set = self._get_name_set(input_names)
        if not name_set:
            raise RuntimeError("No intersected names in given inputs.")

        names = sorted(name_set)

        for index, node in enumerate(self._nodes):
            _cls = node.__class__
            _inputs = {}
            for dt in _cls.input_types:
                if dt.key in inputs:
                    if not inputs[dt.key].exists():
                        raise RuntimeError(f"Input {inputs[dt.key]!r} doesn't exist")
                    _inputs[dt.key] = [
                        path
                        for name, path in zip(input_names[dt.key], paths[dt.key])
                        if name in name_set
                    ]
                else:
                    _inputs[dt.key] = self._get_paths(dt, names)

            _outputs = {dt.key: self._get_paths(dt, names) for dt in _cls.output_types}
            node.init(_inputs, _outputs)

    def execute(
        self,
        verbose: bool = False,
        chunk_size: Optional[int] = None,
        test_run: bool = True,
    ) -> List[NodeExecReport]:
        if not self.is_initialized():
            raise RuntimeError("One or more nodes are not initialized yet.")

        reports = []
        generators = [node(verbose, chunk_size, test_run) for node in self._nodes]
        try:
            while True:
                for gen in generators:
                    report = next(gen)
                    reports.append(report)

        except StopIteration:
            pass

        return reports

    def is_initialized(self) -> bool:
        return self._nodes is not None and all(
            (node.is_initialized() for node in self._nodes)
        )

    def _set_nodes(self, nodes: Union[Path, List[Dict[str, Any]]]) -> None:
        if isinstance(nodes, Path):
            nodes = self._read_nodes(nodes)

        node_classes = get_node_classes()
        node_params = [(node["name"], node["params"]) for node in nodes]
        self._nodes = [node_classes[name](**params) for name, params in node_params]

    def _read_nodes(self, path: Path) -> List[Dict[str, Any]]:
        with open(path) as ifile:
            nodes = json.load(ifile)
        return nodes

    def _get_paths(self, data_type: DataType, names: List[str]) -> List[Path]:
        return [data_type.get_path(self.output_dir, name) for name in names]

    def _make_name(self, path: Path, root: Path) -> str:
        return "_".join(path.with_suffix("").relative_to(root).parts)

    def _deduce_names(
        self, inputs: Dict[str, Path], paths: Dict[str, List[Path]]
    ) -> Dict[str, List[str]]:
        return {
            key: [self._make_name(path, root) for path in paths[key]]
            for key, root in inputs.items()
        }

    def _get_name_set(self, names: Dict[str, List[str]]) -> Set[str]:
        return set.intersection(*map(set, names.values()))
