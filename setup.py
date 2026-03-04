from setuptools import setup, find_packages

setup(
    name="csi-sentinel",
    version="5.0.0",
    author="Ujjwal Manot",
    author_email="ujjwal.manot@example.com",
    description="Non-Line-of-Sight Semantic Wireless Sensing using Wi-Fi CSI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ujjwalmanot/CSI-Sentinel-v5.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
        "PyYAML>=6.0",
        "h5py>=3.8.0",
        "tqdm>=4.65.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "csi-sentinel=main:main",
            "csi-train=training.train_wiclip:main",
            "csi-dashboard=ui.dashboard:main",
        ],
    },
)
