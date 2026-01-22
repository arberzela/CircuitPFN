from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="attn-scm",
    version="0.1.0",
    author="Arber Zela",
    author_email="arber.zela@tue.ellis.eu",
    description="Zero-Shot Causal Graph Extraction from Tabular Foundation Models via Attention Map Decoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/attn-scm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
