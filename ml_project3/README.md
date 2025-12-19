# UiO Machine Learning
 
 ## Setup
The repository contains three machine learning projects:

### [Project1](./ml_project1/)
Linear Regression, Optimization and Bias-Variance tradeoff
### [Project2](./ml_project2/)
Neural Networks for regression and classification
### [Project3](./ml_project3/)
Feature selection and LSTM modeling of snowpack evolution

## Installation
To download the repository in a local folder, open the terminal in your folder and run:

`git clone https://github.com/p-perrone/UiO_MachineLearning`

Inside `../UiO_MachineLearning`, there are three branches `ml_project1`, `ml_project2`, `ml_project3`.
To install the modeules from a specific project folder, navigate to the cloned folder `../UiO_MachineLearning/` and type:

`cd ml_project<replace with the desired project number>`

`pip install .`

This will install the module `functions.py` that is inside every project branch.

To use this module, run in a Python environment:

`import ml_project<replace with the desired project number>.functions`

