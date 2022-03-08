from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(name="protclf",
      url="https://github.com/mihaimorariu/protclf",
      install_requires=required,
      packages=["protclf", "protclf.dataset", "protclf.model"])
