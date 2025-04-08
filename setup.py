from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sirv-integration-pipeline",
    version="0.1.0",
    author="Manveer Chauhan",
    author_email="mschauhan@student.unimelb.edu.au",
    description="A pipeline for integrating SIRV reads into scRNA-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manveerchauhan/sirv-integration-pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pysam>=0.16.0",
        "biopython>=1.78",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jinja2>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "sirv-pipeline=sirv_pipeline.__main__:main",
        ],
    },
)