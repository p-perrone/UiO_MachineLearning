from setuptools import setup, find_packages

setup(
    name="ml-project1",
    version="0.1.0",
    packages=find_packages(),
    description="UiO Machine Learning projects",
    package_dir={'': '..'},  # Look for packages in parent directory
    py_modules=['ml_project1'],
)
