import collections
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dpl.processor.nodes import get_node_classes
from dpl.processor.nodes.base import BaseNode, NodeExecReport
from dpl.processor.datatype import DataType
from dpl import common


class Engine:
    def __init__(self, nodes: Union[Path, List[Dict[str, Any]]], output_dir: Path) -> None:
        self._set_nodes(nodes)
        self.output_dir = output_dir

    def init(self, inputs: Dict[str, Path]) -> None:
        data_types = {dt.key: dt for dt in DataType}
        paths = {
            key: common.listdir(
                path, ext=data_types[key].extensions(), recursive=True,
            )
            for key, path in inputs.items()
        }
        names = self._deduce_names(inputs, paths)
        for index, node in enumerate(self._nodes):
            _cls = node.__class__
            _inputs = self._make_paths(names, _cls.input_types)
            for key in _inputs:
                if key in inputs:
                    _inputs[key] = paths[key][:]
            _outputs = self._make_paths(names, _cls.output_types)
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

    def _make_paths(
        self, names: List[str], data_types: List[DataType]
    ) -> Dict[str, List[Path]]:
        paths: Dict[str, List[Path]] = {}
        for dt in data_types:
            template = dt.template()
            paths[dt.key] = [
                Path(template.format(root=self.output_dir, name=name)) for name in names
            ]
        return paths

    def _deduce_names(
        self, inputs: Dict[str, Path], paths: Dict[str, List[Path]]
    ) -> List[str]:
        if not inputs:
            return []
        first_key = list(inputs.keys())[0]
        root = inputs[first_key]
        return [
            "_".join(path.with_suffix("").relative_to(root).parts)
            for path in paths[first_key]
        ]
