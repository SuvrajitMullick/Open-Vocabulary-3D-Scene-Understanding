from setuptools import setup, find_packages

setup(
    name="sam_hq",
    version="1.0",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "all": [
            "matplotlib==3.8.3",
            "pycocotools==2.0.7",
            "opencv-python==4.9.0.80",
            "onnx==1.15.0",
            "onnxruntime==1.17.1",
            "timm==0.9.16",
        ],
    },
)
