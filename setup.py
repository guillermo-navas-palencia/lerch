import ast
import io
import os
import platform
import re
import sys

from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand


setup(
    name="lerch",
    version='0.1.0',
    description='Lerch transcendent for arbitrary-precision',
    author='Guillermo Navas-Palencia',
    author_email='g.navas.palencia@gmail.com',
    packages=find_packages(),
    platforms='any',
    include_package_data=True,
    license='LGPL'
    )