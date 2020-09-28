from os import path

from setuptools import find_packages, setup

# Define project root folder for later refs
ROOT = path.abspath(path.dirname(__file__))

# Read in the README.md to form the long description
with open(path.join(ROOT, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="""Using machine learning to understanding relationships between
    dynamic structural response and thermomechanical couple loads.""",
    author="Tom Fleet",
    license="BSD-3",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
