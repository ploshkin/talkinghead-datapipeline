#! /usr/bin/zsh

ROOT=$(git rev-parse --show-toplevel)

python3.9 -m venv $ROOT/venvs/dpl
source $ROOT/venvs/dpl/bin/activate

pip install -r $ROOT/requirements/dpl.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html
