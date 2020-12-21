from setuptools import setup, find_packages

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='kyupy',
    version='0.0.2',
    description='High-performance processing and analysis of non-hierarchical VLSI designs',
    long_description=long_description,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url='https://github.com/s-holst/kyupy',
    author='Stefan Holst',
    author_email='mail@s-holst.de',
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17.0',
        'lark-parser>=0.8.0'
    ],
    extras_requires={
        'dev': [
            'pytest>=6.1',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
