# Build recipe only: not yet built or validated on T4.
FROM mambaorg/micromamba:2.3.2 AS micromamba
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
ENV DEBIAN_FRONTEND=noninteractive MAMBA_ROOT_PREFIX=/opt/micromamba
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN micromamba create -y -p /opt/envs/detector -c conda-forge python=3.9 pip && micromamba create -y -p /opt/envs/refinement -c conda-forge python=3.11 pip && micromamba clean --all --yes
WORKDIR /opt/algorithm
COPY environment/requirements-detector.txt environment/requirements-refinement.txt ./environment/
RUN /opt/envs/detector/bin/python -m pip install --no-cache-dir -r environment/requirements-detector.txt && /opt/envs/refinement/bin/python -m pip install --no-cache-dir -r environment/requirements-refinement.txt
COPY . /opt/algorithm
# Explicit T4 (sm75), V100 (sm70), and forward-compatible PTX. No GPU required to compile.
RUN TORCH_CUDA_ARCH_LIST="7.0;7.5+PTX" MAX_JOBS=2 /opt/envs/detector/bin/python environment/build_nndet_extension.py build_ext --inplace
ENV TASK2_DETECTOR_PYTHON=/opt/envs/detector/bin/python TASK2_REFINEMENT_PYTHON=/opt/envs/refinement/bin/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN useradd -m -u 1000 algorithm && mkdir -p /output && chown algorithm:algorithm /output
USER algorithm
ENTRYPOINT ["/opt/envs/refinement/bin/python", "/opt/algorithm/inference.py"]
