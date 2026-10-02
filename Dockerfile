# syntax=docker/dockerfile:1
#
# Strata: Qwen3.8-Flash-Next on NVIDIA GPUs - Tesla P100 (sm_60) community build.
#
# This fork targets the NVIDIA Tesla P100 (compute capability 6.0, Pascal) with a
# CUDA 12.4 / Ubuntu 22.04 base, matching the PrismML project's validated stack.
# CUDA 13 dropped offline compilation for sm_60, so 12.4 is the last usable toolkit.
#
# The engine is compiled during docker build, so the first container start only
# downloads the model (~70 GB) and starts the server. docker build has no GPU,
# so the CUDA architectures are fixed here instead of read from nvidia-smi: the
# engine is a fat binary with a cubin per listed arch, and the runtime picks the
# one matching your card. Narrow CUDA_ARCHITECTURES to your card for a faster
# build; a card outside the set needs a rebuild with its own arch.
#
# Build:
#   docker build -t strata-p100 .
#
# Run (host needs an NVIDIA driver >= 535 and nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -p 8080:8080 \
#     --ulimit memlock=-1 \
#     -v strata-data:/data \
#     -e MODEL=IQ2_XS \
#     strata-p100
#
# Setup choices are env vars, read by docker-entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image. Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.

# 【修改1】基础镜像对齐 PrismML 已验证的 CUDA 12.4 + Ubuntu 22.04 基线
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# 【修改2】默认架构仅保留 P100 (sm_60)。
# CUDA 12.4 是最后一个支持 Pascal 离线编译的工具链。
# 注意：CMAKE_CUDA_ARCHITECTURES 的默认值改为 60，配合下面传入的
# -DSTRATA_EXPERIMENTAL_SM60=ON，CMakeLists 会放行 sm_60 的编译。
ARG CUDA_ARCHITECTURES=60
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine and the image encoder, built
# exactly the way setup.py builds them. BUILD.json is what setup.py reads to
# decide whether an engine is current: source=local with a matching src hash
# means the first start reuses it instead of recompiling.
#
# 【修改3】编译阶段加入 -DSTRATA_EXPERIMENTAL_SM60=ON。
# 这是 Strata 官方为 Pascal (sm_60) / Volta (sm_70) 准备的社区构建开关：
#   * 跳过 CMakeLists 中"拒绝低于 75"的 FATAL_ERROR 检查；
#   * 让 device.cu 的运行时设备检查接受 sm_60；
#   * 启用 include/strata/kernels/dp4a.hpp 中针对 sm_60 的
#     __dp4a (6.1+) 与 __nanosleep (7.0+) 回退实现。
# 官方注释明确说明：community-tested, not in the ready-made engine。
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
arch = os.environ.get("CUDA_ARCHITECTURES", "60").strip().strip('"').replace(",", ";")
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     "-DSTRATA_EXPERIMENTAL_SM60=ON",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")
if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
        [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
         "-DSTRATA_EXPERIMENTAL_SM60=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"], None, "build-vision.bat")

eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
bindir = pathlib.Path(nvcc).parent
meta = {"source": "local", "version": setup.source_version(),
        "archs": [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()], "vision": vision,
        "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs
RUN rm -rf build build-vision

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
