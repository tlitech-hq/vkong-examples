#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export MAX_JOBS="${MAX_JOBS:-4}"
apt-get update
apt-get install -y --no-install-recommends git build-essential libjpeg-dev libgl1 libglib2.0-0
if [ ! -d upstream/.git ]; then
    git init upstream
    git -C upstream remote add origin https://github.com/microsoft/TRELLIS.2.git
fi
if ! git -C upstream cat-file -e 75fbf0183001ed9876c8dbb35de6b68552ee08bd 2>/dev/null; then
    git -C upstream fetch --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd
fi
git -C upstream checkout --detach 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C upstream submodule update --init --recursive --depth 1
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.3 --no-build-isolation
# Pin the CUDA extensions that upstream setup.sh otherwise clones from main.
for spec in CuMesh:12289e1062f0603f2f0d0771b02e1395d247f26f FlexGEMM:6dd94a859c26ee8246888502eada3dd8ad85532e; do
    repo="${spec%%:*}"
    revision="${spec#*:}"
    if [ ! -d "upstream/extensions/$repo/.git" ]; then
        git clone "https://github.com/JeffreyXiang/$repo.git" "upstream/extensions/$repo"
    fi
    git -C "upstream/extensions/$repo" checkout "$revision"
    git -C "upstream/extensions/$repo" submodule update --init --recursive
    module=cumesh
    if [ "$repo" = FlexGEMM ]; then module=flex_gemm; fi
    if ! python -c "import $module"; then
        python -m pip install "./upstream/extensions/$repo" --no-build-isolation
    fi
done
if ! python -c 'import o_voxel'; then
    python -m pip install ./upstream/o-voxel --no-build-isolation --no-deps
fi
if ! python -c 'import nvdiffrast'; then
    python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0 --no-build-isolation
fi
if ! python -c 'import nvdiffrec_render'; then
    python -m pip install git+https://github.com/JeffreyXiang/nvdiffrec.git@b296927cc7fd01c2ac1087c8065c4d7248f72da4 --no-build-isolation
fi

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
unset HF_HUB_ENABLE_HF_TRANSFER
python -u prepare.py
