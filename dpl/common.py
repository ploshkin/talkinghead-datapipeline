from pathlib import Path
from typing import List, Optional


def listdir(path: Path, ext: Optional[List[str]] = None) -> List[Path]:
    if ext:
        return sorted((p for p in path.iterdir() if p.suffix in ext))
    return sorted(path.iterdir())
