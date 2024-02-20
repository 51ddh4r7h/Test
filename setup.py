from setuptools import setup, find_packages

setup(
    name='MLOps-V-0.1',
    version='1.0.0',
    packages=find_packages(where='src'), 
    package_dir={'': 'src'}, 
    install_requires=[
        'tensorflow>=2.5.0'
    ], 
)
