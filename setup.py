from setuptools import setup, find_packages

setup(name='torcheras', 
    install_requires=['numpy>=1.9.1',
                      'torch>=0.4.0'],
    version='0.2.0',
    author="Harry Xie", 
    packages=find_packages())
