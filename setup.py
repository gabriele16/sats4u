import subprocess
import sys, os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# # Get the root directory path (one level above setup.py)
# root_dir = os.path.dirname(os.path.abspath(__file__))

# print(root_dir)
# # Fetch all files in the 'data' directory that end with 'data.csv'
# data_files = glob.glob(os.path.join(root_dir, 'data', '*data.csv'))

# print(data_files)

# # Create a list of tuples with destination directory and file paths
# data_files = [('data', [os.path.relpath(file_path, root_dir)]) for file_path in data_files]

setup(
    name='sats4u',
    version='0.1.0',
    author='Gabriele Tocci',
    description='Workflow to trade, analyse and accumulate satoshis and other crypto currencies',
    packages=find_packages(),
    install_requires=requirements,        
    entry_points={
        'console_scripts': [
            'sats4u-liveapp=src.sats4ulive:main',
        ],
    },
    package_data={'': ['data/*.csv']},
    include_package_data=True,
)