from setuptools import setup, find_packages

setup(
    name="ml_project1",
    version="0.1",
    packages=find_packages(),
    package_dir={'': '.'},  # Important for subdirectory setup
    description="ML Project 1 - Function utilities for machine learning",
    author="Your Name",
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.0',
        'scikit-learn>=0.24',
    ],
)
