import collections
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dpl.processor.nodes import get_node_classes
from dpl.processor.nodes.base import BaseNode, NodeExecReport
from dpl.processor.datatype import DataType


class Engine:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._nodes = None

    @classmethod
    def from_config(cls, path: Path) -> "Engine":
        with open(path) as ifile:
            config = json.load(ifile)

        engine = cls(config["cache_dir"])
        node_params = [(node["name"], node["params"]) for node in config["nodes"]]
        engine.set_nodes(node_params)
        return engine

    def set_nodes(self, node_params: List[Tuple[str, Dict[str, Any]]]) -> None:
        node_classes = get_node_classes()
        self._nodes = [node_classes[name](**params) for name, params in node_params]

    def init(self, input_root: Path, inputs: Dict[str, List[Path]]) -> None:
        names = self._deduce_names_from_inputs(input_root, inputs)
        for index, node in enumerate(self._nodes):
            _cls = node.__class__
            _inputs = self._make_paths(names, _cls.input_types)
            for key in _inputs:
                if key in inputs:
                    _inputs[key] = inputs[key][:]
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

    def _make_paths(
        self, names: List[str], data_types: List[DataType]
    ) -> Dict[str, List[Path]]:
        paths: Dict[str, List[Path]] = {}
        for dt in data_types:
            template = dt.template()
            paths[dt.key] = [
                Path(template.format(root=self.cache_dir, name=name)) for name in names
            ]
        return paths

    def _deduce_names_from_inputs(
        self, root: Path, inputs: Dict[str, List[Path]]
    ) -> List[str]:
        if not inputs:
            return []
        first_key = list(inputs.keys())[0]
        return [
            "_".join(path.with_suffix("").relative_to(root).parts)
            for path in inputs[first_key]
        ]

    def is_initialized(self) -> bool:
        return self._nodes is not None and all(
            (node.is_initialized() for node in self._nodes)
        )
