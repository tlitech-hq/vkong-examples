#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS="${MAX_JOBS:-4}"
apt-get update
apt-get install -y --no-install-recommends git build-essential python3-dev libgl1 libglib2.0-0 libegl1 libxrender1 libxi6 libxkbcommon0 libsm6
if [ ! -d upstream/.git ]; then
    git init upstream
    git -C upstream remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
fi
if ! git -C upstream cat-file -e 82920d643c0dc2f7bfd7255f45f62d386edfe60c 2>/dev/null; then
    git -C upstream fetch --depth 1 origin 82920d643c0dc2f7bfd7255f45f62d386edfe60c
fi
git -C upstream checkout --detach 82920d643c0dc2f7bfd7255f45f62d386edfe60c
python -m pip install -r requirements.txt
python -m pip install ./upstream/hy3dpaint/custom_rasterizer --no-build-isolation
(cd upstream/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh)
mkdir -p upstream/hy3dpaint/ckpt
if [ ! -s upstream/hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]; then
    python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
path = Path('upstream/hy3dpaint/ckpt/RealESRGAN_x4plus.pth')
temporary = path.with_suffix('.incomplete')
urlretrieve('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', temporary)
temporary.replace(path)
PY
fi
