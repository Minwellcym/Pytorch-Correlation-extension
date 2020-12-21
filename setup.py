import os
from setuptools import setup

import torch
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension, CppExtension

project_root = 'Correlation_Module'

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"

source_files = ['correlation.cpp', 'correlation_sampler.cpp']

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_extension():
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        if torch_ver < [1, 7]:
            # supported by https://github.com/pytorch/pytorch/pull/43931
            CC = os.environ.get("CC", None)
            if CC is not None:
                extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    sources = [os.path.join(project_root, file) for file in source_files]
    ext_modules = [
        extension(
            'spatial_correlation_sampler_backend',
            sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=['-lgomp'])
        ]
    return ext_modules


def launch_setup():

    setup(
        name='spatial_correlation_sampler',
        version="0.3.0",
        author="Clément Pinard",
        author_email="clement.pinard@ensta-paristech.fr",
        description="Correlation module for pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ClementPinard/Pytorch-Correlation-extension",
        install_requires=['torch>=1.1', 'numpy'],
        ext_modules=get_extension(),
        package_dir={'': project_root},
        packages=['spatial_correlation_sampler'],
        cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ])


if __name__ == '__main__':
    launch_setup()
