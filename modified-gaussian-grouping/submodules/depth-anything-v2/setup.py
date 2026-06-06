from setuptools import setup, find_packages

setup(
    name="depth_anything_v2",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "all": [
            "gradio_imageslider",
            "gradio==4.29.0",
            "matplotlib",
            "opencv-python",
            "torch",
            "torchvision",
        ],
    },
)
