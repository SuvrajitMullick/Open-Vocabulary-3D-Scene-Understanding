# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="sam_langsplat",
    version="1.0",
    install_requires=[],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib==3.8.3", "pycocotools==2.0.7", "opencv-python==4.9.0.80", "onnx==1.15.0", "onnxruntime==1.17.1"],
        "dev": ["flake8==7.0.0", "isort==5.13.2", "black==24.2.0", "mypy==1.8.0"],
    },
)
