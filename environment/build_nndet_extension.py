from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
root=Path(__file__).resolve().parents[1]/"source/dependencies"
csrc=root/"nndet/csrc"
import os
os.chdir(root)
setup(name="task2-nndet-extension", ext_modules=[CUDAExtension("nndet._C", [str(csrc/"ops.cpp"),str(csrc/"cuda/nms.cu")], extra_compile_args={"cxx":["-O3"],"nvcc":["-O3"]})],cmdclass={"build_ext":BuildExtension})
