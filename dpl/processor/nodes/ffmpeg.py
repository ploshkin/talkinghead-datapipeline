import abc
from pathlib import Path
import subprocess as sp
from typing import Dict, Callable

from joblib import Parallel, delayed
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, NodeExecReport


FFMPEG_CONVERT_CMD = """ffmpeg \\
-hide_banner -loglevel panic -nostats \\
-i {source} {target} -y"""


FFMPEG_TO_IMG_CMD = """ffmpeg \\
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
    input_keys = []
    output_keys = []

    def __init__(self, num_jobs: int = 32) -> None:
        super().__init__()
        self.num_jobs = num_jobs

    def __call__(self, verbose: bool = False) -> NodeExecReport:
        name = self.__class__.__name__

        if self.is_base():
            raise RuntimeError(f"This is instance of the base class: {name}")

        if not self.is_initialized():
            raise RuntimeError("Inputs and outputs are not specified yet.")

        report = NodeExecReport.no_information(name, total=len(self))

        input_key = self.__class__.input_keys[0]
        output_key = self.__class__.output_keys[0]
        iterator = zip(self.inputs[input_key], self.outputs[output_key])
        if verbose:
            iterator = tqdm(iterator, desc=name, total=len(self))

        convert_fn = self.get_convert_fn()
        with Parallel(n_jobs=self.num_jobs, prefer="processes") as parallel:
            parallel(delayed(convert_fn)(src, dst) for src, dst in iterator)

        return report

    def get_convert_fn(self) -> Callable[[Path, Path], None]:
        global convert
        return convert

    def is_base(self) -> bool:
        return not bool(self.__class__.input_keys) and not bool(
            self.__class__.output_keys
        )


class VideoToImagesNode(FfmpegBaseNode):
    input_keys = ["video"]
    output_keys = ["images"]

    def __init__(self, ext: str = ".jpg", num_jobs: int = 32) -> None:
        super().__init__(num_jobs)
        self.ext = ext

    def get_convert_fn(self) -> Callable[[Path, Path], None]:
        ext = self.ext
        global convert_video_to_images

        def convert_fn(source: Path, target: Path) -> None:
            convert_video_to_images(source, target, ext)

        return convert_fn


class VideoToWavNode(FfmpegBaseNode):
    input_keys = ["video"]
    output_keys = ["wav"]


class AacToWavNode(FfmpegBaseNode):
    input_keys = ["aac"]
    output_keys = ["wav"]
