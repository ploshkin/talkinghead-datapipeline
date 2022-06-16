import collections
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dpl.processor.nodes import get_node_classes
from dpl.processor.nodes.base import BaseNode, NodeExecReport


class Engine:
    def __init__(
        self,
        cache_dir: Path,
        path_templates: Dict[str, str],
        verbose: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.path_templates = path_templates
        self.verbose = verbose

        self._nodes = None

    @classmethod
    def from_config(cls, path: Path) -> "Engine":
        with open(path) as ifile:
            config = json.load(ifile)

        engine = cls(config["cache_dir"], config["path_templates"], config["verbose"])
        node_params = [(node["name"], node["params"]) for node in config["nodes"]]
        engine.set_nodes(node_params)
        return engine

    def set_nodes(self, node_params: List[Tuple[str, Dict[str, Any]]]) -> None:
        node_classes = get_node_classes()
        self._nodes = [node_classes[name](**params) for name, params in node_params]
        self._check_path_templates()

    def init(self, input_root: Path, inputs: Dict[str, List[Path]]) -> None:
        names = self._deduce_names_from_inputs(input_root, inputs)
        for index, node in enumerate(self._nodes):
            _cls = node.__class__
            _inputs = self._make_paths(names, _cls.input_keys)
            for key in _inputs:
                if key in inputs:
                    _inputs[key] = inputs[key][:]
            _outputs = self._make_paths(names, _cls.output_keys)
            node.init(_inputs, _outputs)

    def execute(self) -> List[NodeExecReport]:
        if not self.is_initialized():
            raise RuntimeError("One or more nodes are not initialized yet.")

        reports = [node(self.verbose) for node in self._nodes]
        return reports

    def _check_path_templates(self) -> None:
        classes = [node.__class__ for node in self._nodes]
        keys = [_cls.input_keys + _cls.output_keys for _cls in classes]
        keys = set(itertools.chain.from_iterable(keys))

        missing_keys = list(keys - set(self.path_templates.keys()))
        if missing_keys:
            raise RuntimeError(
                f"Template paths are not specified for keys: {missing_keys}"
            )

    def _make_paths(self, names: List[str], keys: List[str]) -> Dict[str, List[Path]]:
        paths: Dict[str, List[Path]] = {}
        for key in keys:
            template = self.path_templates[key]
            paths[key] = [
                Path(template.format(root=self.cache_dir, key=key, name=name))
                for name in names
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
