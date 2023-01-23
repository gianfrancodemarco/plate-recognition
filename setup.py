from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements_app.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]


setup(
    name='src',
    packages=["src"],
    version='0.1.0',
    description='A Python ML boilerplate based on Cookiecutter Data Science, providing support for data versioning (DVC), experiment tracking, Model&Dataset cards, etc.',
    author='Gianfranco Demarco',
    license='',
    python_requires=">=3.8",
    install_requires=[required_packages]
)
