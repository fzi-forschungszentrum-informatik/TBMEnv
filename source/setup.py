#!/usr/bin/env python3
"""
"""
from setuptools import setup, find_packages
from distutils.util import convert_path

# Fetch version from file as suggested here:
# https://stackoverflow.com/a/24517154
main_ns = {}
ver_path = convert_path("tbmenv/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="TBMEnv",
    version=main_ns["__version__"],
    author="David Wölfle",
    author_email="woelfle@fzi.de",
    url="https://github.com/fzi-forschungszentrum-informatik/TBMEnv",
    packages=find_packages(),
    package_data={"": ["*.bz2", "*.json"]},
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "scipy",
    ],
    extras_require={
        "baselines": [
            "ray[tune]",
            "HEBO==0.3.5",
        ],
    },
)
