from setuptools import setup, find_packages

setup(name='torcheras', 
    install_requires=['numpy>=1.9.1',
                      'torch>=0.4.0',
                      'flask'],
    version='0.3.0',
    author="Harry Xie", 
    packages=find_packages(),
    include_package_data = True,
    entry_points = {
        'console_scripts': ['torcheras = server.__main__:main']
    })
