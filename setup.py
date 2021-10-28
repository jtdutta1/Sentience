import sentience
from setuptools import setup, find_packages

setup(name="sentience",
      version=sentience.__version__,
      packages=find_packages(),
      install_requires=["numpy"])