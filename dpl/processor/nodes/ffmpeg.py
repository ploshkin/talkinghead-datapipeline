import abc
from pathlib import Path
import subprocess as sp
from typing import Callable, Dict, Iterable, List

from joblib import Parallel, delayed
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode
from dpl.processor.datatype import DataType


FFMPEG_CONVERT_CMD = """< /dev/null ffmpeg \\
-hide_banner -loglevel panic -nostats \\
-i {source} {target} -y"""


FFMPEG_TO_IMG_CMD = """< /dev/null ffmpeg \\
-hide_banner -loglevel panic -nostats \\
-i {source} -start_number 0 -qscale:v 3 \\
{target}/%6d{ext} -y"""


def convert(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = FFMPEG_CONVERT_CMD.format(source=str(source), target=str(target))
    sp.run(command, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)


def convert_video_to_images(source: Path, target: Path, ext: str = ".jpg") -> None:
    target.mkdir(parents=True, exist_ok=True)
    command = FFMPEG_TO_IMG_CMD.format(source=str(source), target=str(target), ext=ext)
    sp.run(command, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)


class FfmpegBaseNode(BaseNode):
    input_types = []
    output_types = []

    def __init__(self, num_jobs: int = 32, recompute: bool = False) -> None:
        super().__init__(recompute)
        self.num_jobs = num_jobs

        if self.__class__.__name__ != "FfmpegBaseNode":
            self.check_input_output()

    def run_sequence(self, start: int, num: int, verbose: bool) -> None:
        name = self.__class__.__name__

        indices = self._choose_indices_to_process(range(start, start + num))

        input_key = self.input_types[0].key
        output_key = self.output_types[0].key

        iterator = zip(
            (self.inputs[input_key][index] for index in indices),
            (self.outputs[output_key][index] for index in indices),
        )
        if verbose:
            desc = self.get_description(start, num)
            iterator = tqdm(iterator, desc=desc, total=len(indices))

        convert_fn = self.get_convert_fn()
        with Parallel(n_jobs=self.num_jobs, prefer="processes") as parallel:
            parallel(delayed(convert_fn)(src, dst) for src, dst in iterator)

    def get_convert_fn(self) -> Callable[[Path, Path], None]:
        return convert

    def check_input_output(self) -> None:
        if len(self.input_types) != 1:
            raise RuntimeError(
                f"There should be exactly 1 input type, got {len(self.input_types)} "
                f"({self.__class__.__name__})"
            )
        if len(self.output_types) != 1:
            raise RuntimeError(
                f"There should be exactly 1 output type, got {len(self.output_types)} "
                f"({self.__class__.__name__})"
            )

    def _choose_indices_to_process(self, indices: Iterable[int]) -> List[int]:
        indices_to_process = []
        for index in indices:
            input_paths = {
                dt.key: self.inputs[dt.key][index] for dt in self.input_types
            }
            output_paths = {
                dt.key: self.outputs[dt.key][index] for dt in self.output_types
            }

            input_exists = self.check_exist(input_paths)
            need_computation = self.recompute or not self.check_exist(output_paths)

            if input_exists and need_computation:
                indices_to_process.append(index)

        return indices_to_process


class VideoToImagesNode(FfmpegBaseNode):
    input_types = [DataType.VIDEO]
    output_types = [DataType.IMAGES]

    def __init__(
        self, ext: str = ".jpg", num_jobs: int = 32, recompute: bool = False
    ) -> None:
        super().__init__(num_jobs, recompute)
        self.ext = ext

    def get_convert_fn(self) -> Callable[[Path, Path], None]:
        ext = self.ext

        def convert_fn(source: Path, target: Path) -> None:
            convert_video_to_images(source, target, ext)

        return convert_fn


class VideoToWavNode(FfmpegBaseNode):
    input_types = [DataType.VIDEO]
    output_types = [DataType.WAV]


class AacToWavNode(FfmpegBaseNode):
    input_types = [DataType.AAC]
    output_types = [DataType.WAV]
