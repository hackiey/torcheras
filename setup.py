from setuptools import setup, find_packages

setup(name='torcheras', 
    install_requires=['numpy>=1.9.1',
                      'torch>=0.3.0',
                      'visualdl>=0.0.2'],
    version='0.1',
    author="Harry Xie", 
    packages=find_packages())
