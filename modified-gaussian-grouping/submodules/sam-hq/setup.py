from setuptools import setup, find_packages

setup(
    name="segment_anything",
    version="1.0",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "all": [
            "matplotlib",
            "pycocotools",
            "opencv-python",
            "onnx",
            "onnxruntime",
            "timm",
        ],
    },
)
