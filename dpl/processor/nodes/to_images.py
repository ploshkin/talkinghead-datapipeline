from pathlib import Path
import subprocess as sp
from typing import Dict

from joblib import Parallel, delayed
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, NodeExecReport


FFMPEG_CONVERT_CMD = """ffmpeg \\
-hide_banner -loglevel panic -nostats \\
-i {source} -start_number 0 -qscale:v 3 \\
{target}/%6d.jpg -y"""


def convert(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    command = FFMPEG_CONVERT_CMD.format(source=str(source), target=str(target))
    sp.run(command, shell=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)


class ToImagesNode(BaseNode):
    input_keys = ["video"]
    output_keys = ["images"]

    def __init__(self, num_jobs: int = 32) -> None:
        super().__init__()
        self.num_jobs = num_jobs

    def __call__(self, verbose: bool = False) -> NodeExecReport:
        if not self.is_initialized():
            raise RuntimeError("Inputs and outputs are not specified yet.")

        name = self.__class__.__name__
        report = NodeExecReport.no_information(name, total=len(self))

        iterator = zip(self.inputs["video"], self.outputs["images"])
        if verbose:
            iterator = tqdm(iterator, desc=name, total=len(self))

        with Parallel(n_jobs=self.num_jobs, prefer="processes") as parallel:
            parallel(delayed(convert)(src, dst) for src, dst in iterator)
        return report
