# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Muhammad Awad

import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class HIPExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class HIPBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if not isinstance(ext, HIPExtension):
            return super().build_extension(ext)

        # Find hipcc
        hipcc = shutil.which("hipcc")
        if not hipcc:
            hipcc = "/opt/rocm/bin/hipcc"
            if not os.path.exists(hipcc):
                hipcc = "/opt/rocm-6.3.1/bin/hipcc"

        if not os.path.exists(hipcc):
            raise RuntimeError(
                "hipcc not found. Please install ROCm or set ROCM_PATH environment variable."
            )

        # Configuration
        source_file = "csrc/bindings.cpp"
        include_dir = "csrc"

        # Get Python includes
        python_include = subprocess.check_output(
            [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
        ).decode().strip()

        # Get pybind11 includes
        try:
            pybind11_include = subprocess.check_output(
                [sys.executable, "-c", "import pybind11; print(pybind11.get_include())"]
            ).decode().strip()
        except:
            raise RuntimeError("pybind11 not found. Install with: pip install pybind11")

        # Build output path
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        output_file = extdir / Path(self.get_ext_filename(ext.name)).name

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            hipcc,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-D__HIP_PLATFORM_AMD__",
            f"-I{include_dir}",
            f"-I{python_include}",
            f"-I{pybind11_include}",
            source_file,
            "-o", str(output_file),
        ]

        print(f"Building GPU-VDB extension (auto-detecting GPU)...")
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"hipcc failed with exit code {result.returncode}")

        print(f"✅ Successfully built {output_file}")


setup(
    name="gpuvdb",
    version="0.1.0",
    author="Muhammad Awad",
    author_email="mawad@duck.com",
    description="GPU-native VDB implementation for AMD GPUs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maawad/gpuvdb",
    packages=["gpuvdb"],
    ext_modules=[HIPExtension("gpuvdb._gpuvdb")],
    cmdclass={"build_ext": HIPBuild},
    install_requires=[
        "numpy>=1.20",
        "pybind11>=2.10",
    ],
    extras_require={
        "dev": ["pytest", "ruff"],
        "viz": ["matplotlib>=3.5"],
        "torch": ["torch>=2.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)
