from setuptools import setup, find_packages
import codecs
import os.path

packages = find_packages(
        where='.',
        include=[
            'pyevu*',
            # 'cli*'
        ]
)

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    # https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='pyevu',
    version=get_version("pyevu/__init__.py"),
    description='Python Extensive Vector Utility, is a utility for working with vectors in python.',
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # 'argcomplete>=1.12.3',
        'pylint>=2.12.2',
        'pyquaternion>=0.9.9',
        'numpy>=1.21.5'
    ],
    python_requires='>=3.9',
    # entry_points={
    #     "console_scripts": [
    #         "pyevu=cli.example:main"
    #     ]
    # }
)