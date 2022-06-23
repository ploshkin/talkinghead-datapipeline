import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

from dpl.processor.nodes.registry import NodeRegistry
from dpl.processor.datatype import DataType


@dataclass
class NodeExecReport:
    name: str
    missing_inputs: List[Dict[str, str]] = field(default_factory=list)
    error_inputs: List[Dict[str, str]] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    def add_error(self, inputs: Dict[str, Path], message: str) -> None:
        self.error_inputs.append({key: str(path) for key, path in inputs.items()})
        self.error_messages.append(message)

    def add_missing(self, inputs: Dict[str, Path]) -> None:
        self.missing_inputs.append({key: str(path) for key, path in inputs.items()})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class BaseResource:
    def __init__(self) -> None:
        self.__loaded = False

    def __getattr__(self, name: str) -> Any:
        if name != "__loaded":
            if not self.is_loaded():
                self.load()
        return self.__getattribute__(name)

    def __enter__(self) -> "BaseResource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.is_loaded():
            self.unload()

    def is_loaded(self) -> bool:
        return self.__loaded

    def load(self) -> None:
        self.__loaded = True

    def unload(self) -> None:
        self.__loaded = False


class EmptyResource(BaseResource):
    pass


class BaseNode(metaclass=NodeRegistry):
    input_types: List[DataType] = []
    output_types: List[DataType] = []

    def __init__(self, recompute: bool = False) -> None:
        self.recompute = recompute

        self.inputs = None
        self.outputs = None
        self.resource = EmptyResource()

        self._length = 0
        self._num_chars = self._get_max_classname_len()

    def init(
        self,
        inputs: Dict[str, List[Path]],
        outputs: Dict[str, List[Path]],
    ) -> None:
        self._check_inputs(inputs)
        self._check_outputs(outputs)

        input_size = len(inputs[list(inputs.keys())[0]])
        output_size = len(outputs[list(outputs.keys())[0]])

        if input_size != output_size:
            raise RuntimeError("Input size doesn't match to output size")

        self._length = input_size

        self.inputs = inputs
        self.outputs = outputs

        self._report = NodeExecReport(self.__class__.__name__)

    def __call__(
        self,
        verbose: bool = False,
        chunk_size: Optional[int] = None,
        test_run: bool = True,
    ) -> None:
        name = self.__class__.__name__
        if not self.is_initialized():
            raise RuntimeError(f"Node {name!r} is not initialized.")

        first = 0
        if test_run:
            yield self.run_sequence(first, 1, verbose)
            first = 1

        if chunk_size is not None:
            for start in range(first, len(self), chunk_size):
                num = min(chunk_size, len(self) - start)
                yield self.run_sequence(start, num, verbose)
        else:
            yield self.run_sequence(first, len(self) - first, verbose)

    def __len__(self) -> int:
        return self._length

    def run_sequence(self, start: int, num: int, verbose: bool) -> None:
        iterator = range(start, start + num)
        if verbose:
            desc = self.get_description(start, num)
            iterator = tqdm(iterator, desc=desc, total=num)

        with self.resource:
            for index in iterator:
                input_paths = {key: self.inputs[key][index] for key in self.inputs}
                output_paths = {key: self.outputs[key][index] for key in self.outputs}

                if self.recompute or not self.check_exist(output_paths):
                    if self.check_exist(input_paths):
                        try:
                            self.run_single(input_paths, output_paths)
                        except RuntimeError as err:
                            self._report.add_error(input_paths, str(err))
                    else:
                        self._report.add_missing(input_paths)

    @abc.abstractmethod
    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        pass

    @property
    def report(self) -> NodeExecReport:
        return self._report

    def is_initialized(self) -> bool:
        return self.inputs is not None and self.outputs is not None

    def check_exist(self, paths: Dict[str, Path]) -> bool:
        return all(path.exists() for _, path in paths.items())

    def get_description(self, start: int, num: int) -> str:
        progress = start / len(self) * 100
        template = f"{{name:{self._num_chars + 1}}} [{progress:5.1f}%]"
        return template.format(name=f"{self.__class__.__name__}:")

    def _check_inputs(self, inputs: Dict[str, List[Path]]) -> None:
        self._check_num_paths(inputs)
        missing, extra = self._get_missing_and_extra(inputs, self.input_types)
        if missing:
            raise RuntimeError(f"These inputs are missing: {missing}")

    def _check_outputs(self, outputs: Dict[str, List[Path]]) -> None:
        self._check_num_paths(outputs)
        missing, extra = self._get_missing_and_extra(outputs, self.output_types)
        if missing:
            raise RuntimeError(f"These inputs are missing: {missing}")

    def _check_num_paths(self, inputs_or_outputs: Dict[str, List[Path]]) -> None:
        if len(inputs_or_outputs) > 1:
            main_key = list(inputs_or_outputs.keys())[0]
            length = len(inputs_or_outputs[main_key])
            for key, paths in inputs_or_outputs.items():
                if len(paths) != length:
                    raise RuntimeError(
                        f"Length of '{key}' doesn't match to length of '{main_key}': "
                        f"{len(paths)} != {length}"
                    )

    def _get_missing_and_extra(
        self,
        inputs_or_outputs: Dict[str, List[Path]],
        expected_types: List[DataType],
    ) -> Tuple[List[str], List[str]]:
        keys = set(inputs_or_outputs.keys())
        expected_keys = set(dt.key for dt in expected_types)

        extra = list(keys - expected_keys)
        missing = list(expected_keys - keys)
        return missing, extra

    def _get_max_classname_len(self) -> int:
        classnames = self.__class__.get_registry().keys()
        return max(map(len, classnames))
