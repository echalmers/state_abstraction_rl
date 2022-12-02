from gettext import install
from setuptools import setup

setup(
    name="minigrid_envs",
    version="0.0.1",
    install_requires=["gym==0.26.0", "minigrid"]
)