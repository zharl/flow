from setuptools import setup, find_packages

setup(
    name='flowdag',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'ipywidgets',
        'autopep8',
        'pydot',
        'IPython'
    ],
    author='Marco Renedo',
    description='A library for processing and visualizing data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/renedoz/flow',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
