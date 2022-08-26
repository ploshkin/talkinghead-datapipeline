import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseType:
    def __init__(self, ext: Union[str, List[str]]) -> None:
        if isinstance(ext, str):
            self.default_ext = ext
            self._exts = {ext}
        elif isinstance(ext, list):
            self.default_ext = ext[0]
            self._exts = set(ext)
        else:
            raise TypeError("'ext' should be str or list of str")

    def template(self, name: str, ext: Optional[str] = None) -> str:
        ext = self._get_ext(ext)
        return f"{{root}}/{name}/{{name}}{ext}"

    def ffmpeg_template(self, name: str, ext: Optional[str] = None) -> str:
        return self.template(name, ext)

    def is_sequential(self) -> bool:
        return False

    def _get_ext(self, ext: Optional[str] = None) -> str:
        ext = self.default_ext if ext is None else ext
        if ext not in self._exts:
            raise ValueError(
                f"Got ext={ext}, but available extensions are {self._exts}"
            )
        return ext


class NumpyFileType(BaseType):
    def __init__(self) -> None:
        super().__init__(ext=".npy")


class FolderType(BaseType):
    def __init__(self, ext: Union[str, List[str]], num_digits: int = 6) -> None:
        super().__init__(ext)
        self.num_digits = num_digits

    def template(self, name: str, ext: Optional[str] = None) -> str:
        self._get_ext(ext)
        return f"{{root}}/{name}/{{name}}"

    def ffmpeg_template(self, name: str, ext: Optional[str] = None) -> str:
        folder = self.template(name)
        ext = self._get_ext(ext)
        return f"{folder}/%{self.num_digits}d{ext}"

    def is_sequential(self) -> bool:
        return True


class DataType(Enum):
    VIDEO = BaseType(".mp4")
    WAV = BaseType(".wav")
    AAC = BaseType(".m4a")
    IMAGES = FolderType([".jpg", ".png"])
    CROPS = FolderType([".jpg", ".png"])
    RENDER_NORMAL = FolderType([".jpg", ".png"])
    RENDER_UV = FolderType([".jpg", ".png"])
    RENDER_MASK = FolderType([".jpg", ".png"])
    RAW_BBOXES = NumpyFileType()
    BBOXES = NumpyFileType()
    LANDMARKS = NumpyFileType()
    LANDMARKS3D = NumpyFileType()
    SHAPE = NumpyFileType()
    TEX = NumpyFileType()
    EXP = NumpyFileType()
    POSE = NumpyFileType()
    CAM = NumpyFileType()
    LIGHT = NumpyFileType()
    VERTS = NumpyFileType()
    WAV2VEC = NumpyFileType()
    VOLUME = NumpyFileType()
    A2EN = BaseType(".npz")
    VID2VID = BaseType(".h5")
    SRC_SEQ = BaseType(".h5")
    VID2VID_AUDIO = BaseType(".h5")

    def __init__(self, data_type: BaseType) -> None:
        self.key = self.name.lower()
        self._data_type = data_type

    def template(self, ext: Optional[str] = None) -> str:
        return self._data_type.template(self.key, ext)

    def get_path(self, root: Path, name: str, ext: Optional[str] = None) -> Path:
        return Path(self.template(ext).format(root=root, name=name))

    def ffmpeg_template(self, ext: Optional[str] = None) -> str:
        return self._data_type.ffmpeg_template(self.key, ext)

    def extensions(self) -> List[str]:
        return list(self._data_type._exts)

    def is_sequential(self) -> bool:
        return self._data_type.is_sequential()
