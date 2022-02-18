from setuptools import setup, find_packages
from pyevu import __version__

packages = find_packages(
        where='.',
        include=[
            'pyevu*',
            # 'cli*'
        ]
)

setup(
    name='pyevu',
    version=__version__,
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
        'pylint>=2.12.2'
    ],
    python_requires='>=3.9',
    # entry_points={
    #     "console_scripts": [
    #         "pyevu=cli.example:main"
    #     ]
    # }
)