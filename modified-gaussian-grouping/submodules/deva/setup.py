from setuptools import setup, find_packages

setup(
    name="deva",
    version="1.0.0",
    python_requires=">=3.10",
    packages=find_packages(include=["deva", "deva.*"]),
    install_requires=[
        "gitpython>=3.1",
        "hickle>=5.0",
        "tensorboard>=2.13",
        "numpy>=1.24",
        "Pillow>=9.5",
        "opencv-python>=4.8",
        "scipy>=1.11.2",
        "pycocotools>=2.0.7",
        "supervision>=0.18",
        "tqdm>=4.66.1",
        "gurobipy>=10.0.3",
        "PuLP>=2.7",
        "gradio>=3.44",
        "gdown>=4.7.1",
        "timm==0.9.16",
        "open-clip-torch==2.24.0",
        "thinplate @ git+https://github.com/cheind/py-thin-plate-spline",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
