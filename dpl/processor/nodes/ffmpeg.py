import abc
from pathlib import Path
import subprocess as sp
from typing import Dict, Callable

from joblib import Parallel, delayed
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, NodeExecReport
from dpl.processor.datatype import DataType


FFMPEG_CONVERT_CMD = """ffmpeg \\
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

    def run_sequence(self, start: int, num: int, verbose: bool) -> NodeExecReport:
        name = self.__class__.__name__
        if self.is_base():
            raise RuntimeError(f"This is instance of the base class: {name}")

        report = NodeExecReport.no_information(name, start, num)

        input_key = self.input_types[0].key
        output_key = self.output_types[0].key
        iterator = zip(
            self.inputs[input_key][start : start + num],
            self.outputs[output_key][start : start + num],
        )
        if verbose:
            desc = self.get_description(start, num)
            iterator = tqdm(iterator, desc=desc, total=num)

        convert_fn = self.get_convert_fn()
        with Parallel(n_jobs=self.num_jobs, prefer="processes") as parallel:
            parallel(delayed(convert_fn)(src, dst) for src, dst in iterator)

        return report

    def get_convert_fn(self) -> Callable[[Path, Path], None]:
        return convert

    def is_base(self) -> bool:
        return not bool(self.input_types) and not bool(self.output_types)


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
