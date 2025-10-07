# python3
# Source code with all the functions for "Applied Data Analysis and Machine Learning" master course
# University of Oslo, Autumn 2025
#---------------------------------
# author: Pietro Perrone
#
""" 
=====================
ml_project1.functions
=====================
This module contains functions for:
    - Generating the Runge function with or without noise
    - Creating a design matrix for polynomial regression
    - Computing optimal parameters for Ordinary Least Squares (OLS) and Ridge (L2) regression
    - Performing linear regression predictions
    - Standard scaling of data
    - Gradient descent optimization with different methods (AdaGrad, RMSprop, Adam)
    - Lasso regression using gradient descent with L1 regularization
"""

#--------------------------------
# dependencies
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.utils import resample
import autograd.numpy as np
from autograd import grad


#--------------------------------
# FIRST PART: PREPROCESSING, OLS AND RIDGE
#--------------------------------


# Runge function
def Runge(x, noise=True, noisescale=1):
    """ Computes Runge function with or without a normal distributed stochastic noise.
        Parameters:
        :: x (array) = input dataset
        :: noise (bool) = adds some noise if True
    """
    if not isinstance(noise, bool):
        raise TypeError(f"`noise` must be boolean (True or False) and not {type(noise)}")

    if noise == True:
        y = ( 1 / (1 + 25 * x**2) ) + np.random.normal(loc=0, scale=noisescale, size=len(x))
    elif noise == False:
        y = ( 1 / (1 + 25 * x**2) )

    return y


# OLS implementation

import numpy as np

class LinearRegression_own:
    """
    Own implementation of linear regression with OLS and Ridge methods and their analytical solutions.
    Includes polynomial feature expansion and fitting.
    
    Based on scikit-learn linear regression workflow.
    """

    def __init__(self, intercept=True):
        self.intercept = intercept


    def polynomial_features(self, x, p):  
        """ Computes the design matrix for linear regression.

            Parameters:
            :: x (array) = input dataset
            :: p (int) = polynomial degree

            ------------------------------------------------------------------
            Returns:
            :: X (matrix) = design matrix
        """
        
        x = np.asarray(x).ravel()
        n = len(x)

        if not isinstance(self.intercept, bool):
            raise TypeError(f"Intercept must be boolean (True or False), not {type(self.intercept)}")

        # composing the matrix (with or without intercept) ~ `transform` by sklearn
        if self.intercept:
            X = np.ones((n, p + 1))
            for j in range(1, p + 1):
                X[:, j] = x ** j
        else:
            X = np.ones((n, p))
            for j in range(p):
                X[:, j] = x ** (j + 1)

        return X
    

    def fit(self, X, y, method='OLS', lbda=0.1):
        """ Fits linear regression model using analytical solution.
        
            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: y (array) = true value to be modeled
            :: method (str) = 'OLS' or 'Ridge'
            :: lbda (scalar) = penalty coefficient for Ridge regression

            ------------------------------------------------------------------
            Returns:
            :: beta (array) = optimal parameters
        """

        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if method == 'OLS':
            beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        elif method == 'Ridge':
            n_features = X.shape[1]
            beta = np.linalg.inv(X.T @ X + lbda * np.eye(n_features)) @ X.T @ y
        else:
            raise ValueError(f"`method` must be 'OLS' or 'Ridge', not {method}")
        
        return beta
    

    def predict(self, X, beta):
        """ Computes the regression curve (predicted y values).

            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: beta (array) = optimal parameters obtained from fit()

            ------------------------------------------------------------------
            Returns:
            :: y_pred (array) = predicted y values
        """
        return np.asarray(X) @ np.asarray(beta)
    


# standard scaling
def standard_scaler(a, centering=True, stdscaling=True):
    """ Scales input array a.

        Parameters:
        :: a (array) = input 1D array
        :: centering (bool) = substracts a.mean()
        :: scaling (bool) = divides by a.std()

        ------------------------------------------------------------------
        Returns:
        :: scaled_a (array) = scaled version of input array
    """
    if not isinstance(a, np.ndarray):
        raise TypeError("'a' must be a 1D array")
    if not isinstance(centering, bool):
        raise TypeError("'centering' must be boolean (True or False)")
    if not isinstance(stdscaling, bool):
        raise TypeError("'stdscaling' must be boolean (True or False)")

    if centering:
        centered = a - a.mean()
    else:
        centered = a

    if stdscaling:
        scaled_a = centered / a.std()
    else:
        scaled_a = centered
        
    return scaled_a


#--------------------------------
# SECOND PART: GRADIENT DESCENT AND LASSO
#--------------------------------


# defining cost function for OLS, Ridge and Lasso
def cost_function(y, X, theta, regression_method='OLS', lbda=0.1):
    """ Cost function for OLS, Ridge or Lasso regression.

        Parameters:
        :: y (array) = true value to be modeled
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
        :: lbda (scalar) = regularization parameter (only for Ridge and Lasso)

        ------------------------------------------------------------------
        Returns:
        :: cost (scalar) = value of the cost function
    """
    y_pred = X @ theta
    residuals = y_pred - y.reshape(-1, 1)
    
    if regression_method == 'OLS':
        return np.sum(residuals**2)
    
    elif regression_method == 'Ridge':
        return np.sum(residuals**2) + lbda * np.sum(theta**2)
    
    elif regression_method == 'Lasso':
        return np.sum(residuals**2) + lbda * np.sum(np.abs(theta))
    
    else:
        raise ValueError(f"`regression_method` must be 'OLS', 'Ridge' or 'Lasso', not {regression_method}")


# analytical computation of gradient
def grad_analytic(X, y, theta, lbda, regression_method='OLS'):
        """Compute gradient analytically.

            Parameters:
            :: X_batch (matrix) = design matrix for the batch
            :: y_batch (array) = true value to be modeled for the batch
            :: theta (array) = current parameters
            :: lbda (scalar) = regularization parameter (only for Ridge and Lasso)
            :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
            
            ------------------------------------------------------------------
            Returns:
            :: gradient (array) = gradient of the cost function
        """

        if not isinstance(regression_method, str):
             raise TypeError(f"`regression_method` must be a string, not {type(regression_method)}")

        # predicted values
        y_pred = X @ theta
        # residuals
        residuals = y_pred - y.reshape(-1, 1)

        if regression_method == 'OLS':
            return (2.0 / len(y)) * (X.T @ residuals)
        elif regression_method == 'Ridge':
            return 2.0 * ((X.T @ residuals) / len(y) + (lbda * theta))
        elif regression_method == 'Lasso':
            return 2.0 * ((X.T @ residuals) / len(y) + lbda * np.sign(theta))
        else:
             raise ValueError(f"`regression_method` must be either 'OLS', 'Ridge' or 'Lasso', not {regression_method}")


# optimal parameters with simple gradient descent
def theta_gd(X, y, eta, regression_method='OLS', lbda=0.1, iterations=200000, converge=1e-8):
    """ Computes optimal parameters for ordinary least squares regression with gradient descent.
        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: eta (scalar) = learning rate
        :: regression_method (str) = 'OLS' or 'Ridge'
        :: lbda (scalar) = regularization parameter (only for Ridge)
        :: iterations (int) = maximum number of iterations
        :: converge (scalar) = convergence criterion
    """
    # error for wrong regression_method input
    if regression_method not in ['OLS', 'Ridge']:
        raise ValueError("regression_method must be 'OLS' or 'Ridge'")

    # initialize theta, from y shape
    n, m = X.shape
    y = np.asarray(y).reshape(-1, 1)
    theta = np.zeros((m, 1), dtype=float)

    for k in range(iterations):
        gradient = grad_analytic(X, y, theta, lbda=lbda, regression_method=regression_method)  # compute gradient with autograd
        theta -= eta * gradient

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e2:  # cap gradient norm (DeepSeek hint)
            gradient = gradient / grad_norm * 1e2

        if grad_norm <= converge:
            print(f"Stop at epsilon = {grad_norm:.2e}, iteration = {k}")
            break

    return theta


# optimal parameters with momentum gradient descent and different eta update methods
def theta_gd_mom(X, y, regression_method='OLS', eta=1e-3, eta_update_method='Simple_momentum', lbdaRidge=0.1, 
                lbdaLasso=0.1, momentum=0.9, iterations=2000, converge = 1e-8):
    """ Computes optimal parameters for ordinary least squares regression with momentum gradient descent.

        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
        :: eta (scalar) = learning rate - default 1e-3 (optimal from experiments results)
        :: lbdaRidge (scalar) = penalty coefficient for Ridge regression
        :: lbdaLasso (scalar) = penalty coefficient for Lasso regression
        :: momentum (scalar) = momentum coefficient for momentum gradient descent
        :: eta_update_method (str) = 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum'
        :: iterations (int) = maximum number of iterations
        :: converge (scalar) = convergence criterion

        ------------------------------------------------------------------
        Returns:
        :: theta (array) = optimal parameters
    """

    # error for wrong input
    if regression_method != 'OLS' and regression_method != 'Ridge' and regression_method != 'Lasso':
        raise ValueError(f"'regression_method' must be either 'OLS' or 'Ridge' and not {regression_method}")
    if eta_update_method not in ['AdaGrad', 'RMSprop', 'ADAM', 'Simple_momentum']:
        raise ValueError(f"'eta_update_method' must be either 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum' and not {eta_update_method}")

    # initialize random theta and velocity
    theta = np.random.randn(X.shape[1], 1)
    velocity = np.zeros((X.shape[1], 1))

    # common params
    delta = 1e-8    # to avoid division by zero
    t = 0           # time step for Adam
    Giter = 0.0     # initialize sum of squared gradients for AdaGrad, RMSprop and Adam

    # defining cost function
    if regression_method == 'OLS':
        cost = lambda theta: cost_function(y, X, theta)
    elif regression_method == 'Ridge':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Ridge', lbda=lbdaRidge)
    elif regression_method == 'Lasso':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Lasso', lbda=lbdaLasso)

    for k in range(iterations):
        gradient = grad(cost)(theta).reshape(-1, 1)  # compute gradient with autograd
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            print(f"NaN in gradient at iteration {k}")
            print("Gradient stats:", np.min(gradient), np.max(gradient))
            break

        # update theta according to the chosen method
        if eta_update_method == 'Simple_momentum':
            velocity = momentum * velocity - eta * gradient     # simple momentum update
            update = velocity
            theta += update
        
        elif eta_update_method == 'AdaGrad': 
            # AdaGrad tracks the sum of the squares of the previous gradients
            Giter += gradient**2    # square the gradient
            adjusted_eta = eta / (delta + np.sqrt(Giter))
            update = adjusted_eta * gradient
            theta -= update

        elif eta_update_method == 'RMSprop':
            # RMSprop modificates AdaGrad by useing a moving average of squared gradients to normalize the gradient
            beta = 0.99     # discount factor that controls the averaging time of the second moment (variance of the gradient)
            Giter = beta * Giter + (1 - beta) * gradient**2
            adjusted_eta = eta / (delta + np.sqrt(Giter))
            update = adjusted_eta * gradient
            theta -= update

        elif eta_update_method == 'ADAM':
            # Adam (Adaptive Moment Estimation) is an extension to RMSprop that also takes 
            # into account the moving average of the gradient itself
            beta1 = 0.9     # decay rate for the first moment estimate (velocity = weighted average of the past gradients)
            beta2 = 0.99    # decay rate for the second moment estimate (Giter = weighted average of the past squared gradients)
            t += 1          # time step
            
            # update first moment estimate
            velocity = beta1 * velocity + (1 - beta1) * gradient
            # update second moment estimate
            Giter = beta2 * Giter + (1 - beta2) * gradient**2
            
            # compute bias-corrected first moment estimate
            velocity_corrected = velocity / (1 - beta1**t)
            # compute bias-corrected second moment estimate
            Giter_corrected = Giter / (1 - beta2**t)

            adjusted_eta = eta / (delta + np.sqrt(Giter_corrected))
            update = adjusted_eta * velocity_corrected
            theta -= update

        # convergence of the algorithm
        update_norm = np.linalg.norm(update)
        if update_norm <= converge:
            print(f"Stop at epsilon = {update_norm:.2e}, iteration = {k}")
            break

    return theta


# Lasso regression with iterative approach
def Lasso_params(X, y, lbda=0.1, iterations=5000, eta=1e-4, converge=1e-8):
    """ Computes optimal parameters for Lasso regression with simple gradient descent.

        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: lbda (scalar) = penalty coefficient
        :: iterations (int) = maximum number of iterations
        :: eta (scalar) = learning rate
        :: converge (scalar) = convergence criterion

        ------------------------------------------------------------------
        Returns:
        :: theta (array) = optimal parameters
    """

    # initialize a zero optimal parameters array in order to avoid numerical instabilities
    theta = np.zeros((X.shape[1], 1))
    
    y = y.reshape(-1,1)

    for k in range(iterations):
        gradient = grad_analytic(X, y, theta, lbda=lbda, regression_method='Lasso')  # compute gradient with autograd
        theta_new = theta - eta * gradient

        # convergence check (proposed by DeepSeek)
        if np.linalg.norm(theta_new - theta) < converge:
            print(f"Lasso converged at iteration {k}")
            break

        theta = theta_new

    return theta


# stochastic now 
def theta_sgd_mom(X, y, regression_method='OLS', eta=1e-3, eta_update_method='Simple_momentum', 
                  lbda=0.1,momentum=0.9, iterations=2000, converge = 1e-8):
    """ Computes optimal parameters for OLS and Ridge regression with momentum stochastic gradient descent.

        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: regression_method (str) = 'OLS' or 'Ridge'
        :: eta (scalar) = learning rate
        :: lbdaRidge (scalar) = penalty coefficient for Ridge regression
        :: lbdaLasso (scalar) = penalty coefficient for Lasso regression
        :: momentum (scalar) = momentum coefficient for momentum gradient descent
        :: eta_update_method (str) = 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum'
        :: iterations (int) = maximum number of iterations
        :: converge (scalar) = convergence criterion

        ------------------------------------------------------------------
        Returns:
        :: theta (array) = optimal parameters
    """

    # error for wrong input
    if regression_method != 'OLS' and regression_method != 'Ridge':
        raise ValueError(f"'regression_method' must be either 'OLS' or 'Ridge' and not {regression_method}")
    if eta_update_method not in ['AdaGrad', 'RMSprop', 'ADAM', 'Simple_momentum']:
        raise ValueError(f"'eta_update_method' must be either 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum' and not {eta_update_method}")

    # initialize random theta and velocity
    theta = np.random.randn(X.shape[1], 1)
    velocity = np.zeros((X.shape[1], 1))

    # common params
    delta = 1e-8    # to avoid division by zero
    t = 0           # time step for Adam
    Giter = 0.0     # initialize sum of squared gradients for AdaGrad, RMSprop and Adam

    # defining cost function
    if regression_method == 'OLS':
        cost = lambda y, X, theta: cost_function(y, X, theta)
    elif regression_method == 'Ridge':
        cost = lambda y, X, theta: cost_function(y, X, theta, regression_method='Ridge', lbda=lbda)

    n_epochs = 50
    mb_size = 5                     #size of each minibatch
    num_mb = int(len(y) / mb_size)  #number of minibatches

    for epoch in range(n_epochs):
        
        for i in range(num_mb):
            mbk_index = mb_size * np.random.randint(num_mb)   # pick random minibatch index
            X_mbk = X[mbk_index : mbk_index + mb_size]      # create kth-minibatch of design matrix
            y_mbk = y[mbk_index : mbk_index + mb_size]      # create kth-minibatch of true values

            gradient = grad_analytic(X_mbk, y_mbk, theta, lbda=lbda, regression_method=regression_method)

            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                print(f"NaN in gradient at iteration {k}")
                print("Gradient stats:", np.min(gradient), np.max(gradient))
                break

            # update theta according to the chosen method
            if eta_update_method == 'Simple_momentum':
                velocity = momentum * velocity - eta * gradient     # simple momentum update
                update = velocity
                theta += update
            
            elif eta_update_method == 'AdaGrad': 
                # AdaGrad tracks the sum of the squares of the previous gradients
                Giter += gradient**2    # square the gradient
                adjusted_eta = eta / (delta + np.sqrt(Giter))
                update = adjusted_eta * gradient
                theta -= update

            elif eta_update_method == 'RMSprop':
                # RMSprop modificates AdaGrad by useing a moving average of squared gradients to normalize the gradient
                beta = 0.99     # discount factor that controls the averaging time of the second moment (variance of the gradient)
                Giter = beta * Giter + (1 - beta) * gradient**2
                adjusted_eta = eta / (delta + np.sqrt(Giter))
                update = adjusted_eta * gradient
                theta -= update

            elif eta_update_method == 'ADAM':
                # Adam (Adaptive Moment Estimation) is an extension to RMSprop that also takes into account the moving average of the gradient itself
                beta1 = 0.9     # decay rate for the first moment estimate (velocity = weighted average of the past gradients)
                beta2 = 0.99    # decay rate for the second moment estimate (Giter = weighted average of the past squared gradients)
                t += 1          # time step
                
                # update first moment estimate
                velocity = beta1 * velocity + (1 - beta1) * gradient
                # update second moment estimate
                Giter = beta2 * Giter + (1 - beta2) * gradient**2
                
                # compute bias-corrected first moment estimate
                velocity_corrected = velocity / (1 - beta1**t)
                # compute bias-corrected second moment estimate
                Giter_corrected = Giter / (1 - beta2**t)

                adjusted_eta = eta / (delta + np.sqrt(Giter_corrected))
                update = adjusted_eta * velocity_corrected
                theta -= update

            # suggested by DeepSeek
            update_norm = np.linalg.norm(update)
            if update_norm > 1000:  # Prevent explosion
                update = update / update_norm * 1000
                update_norm = 1000

            # convergence of the algorithm
            update_norm = np.linalg.norm(update)
            if update_norm <= converge:
                print(f"Stop at epsilon = {update_norm:.2e}, iteration = {i} of epoch {epoch}")
                break

    return theta


#--------------------------------
# THIRD PART: STATISTICAL ANALYSIS - BIAS-VARIANCE TRADEOFF
#--------------------------------

class BiasVarianceTradeoff:
    """ 
    Class to perform bootstrap resampling and bias-variance decomposition.

    Includes bootstrap resampling, bias-variance decompossition, simulation over a range of degree of complexity and plotting.
    See hel() for the specific functions for more details.
    """

    def __init__(self, x, y, p_range=np.arange(0, 30, 1), n_bootstraps=1000):
        self.p_range = p_range
        self.x = x
        self.y = y
        self.n_bootstraps = n_bootstraps

        self.y_pred_matrix = None
        self.y_test = None
        self.mse_array = np.zeros(len(p_range))
        self.bias_array = np.zeros(len(p_range))
        self.variance_array = np.zeros(len(p_range))


    def bootstrap(self, p=5, regression_method='OLS'):
        """ Boostrap resampling for linear regression.

            Parameters:
            :: x (1D array) = input dataset
            :: y (1D array) = true value to be modeled
            :: p (int) = polynomial degree
            :: regression_method (str) = 'OLS' or 'Ridge'
            :: n_bootstraps (int) = number of bootstrap resamplings

            ------------------------------------------------------------------
            Returns:
            :: y_pred_matrix (2D array) = matrix of predicted values from bootstrap resampling

        """
        # error for wrong input
        if regression_method != 'OLS' and regression_method != 'Ridge':
            raise ValueError(f"'regression_method' must be either 'OLS' or 'Ridge' and not {regression_method}")
        
        x_train, x_test, y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
        x_test = scaler.transform(x_test.reshape(-1, 1)).flatten()
        self.y_test = self.y_test.reshape(-1, 1)

        # initialize matrix of predicted values
        self.y_pred_matrix = np.zeros((self.y_test.shape[0], self.n_bootstraps))

        # loop over the number of bootstraps
        for i in range(self.n_bootstraps):
            x_sample, y_sample = resample(x_train, y_train)     # resampling with sklearn

            lr = LinearRegression_own(intercept=True)           # initialize the LinearRegression_own object
            X_train = lr.polynomial_features(x_sample, p)       # transform train set to polynomial features
            X_test = lr.polynomial_features(x_test, p)          # transform test set to polynomial features
            beta = lr.fit(X_train, y_sample, method='OLS')             # fit model to the i-th bootstrap train sample
            y_pred_sample = lr.predict(X_test, beta).ravel()          # predict on the same test set at each i-th iteration

            # update predicted values matrix
            self.y_pred_matrix[:, i] = y_pred_sample
            
        return self.y_pred_matrix
        

    def decompose_mse(self): 
        """ Decomposes the mean squared error (aka total error) into bias and variance.
            The total error is defined as the sum of squared bias, variance and irreducible error:

                        E[(y - y_pred)²] = Bias² + Variance + Irreducible Error

            Parameters:
            :: y_test (1D array) = true value to be modeled
            :: y_pred_matrix (2D array) = matrix of predicted values from bootstrap resampling

            ------------------------------------------------------------------
            Returns:
            :: total error, bias, variance (scalars)
        """

        if self.y_pred_matrix is None:
            raise ValueError("The bootstrap must be run before decomposing MSE (predicted y matrix is None)")

        # compute total error, bias and variance
        # keepdims=True to keep the column vector shape
        mse = np.mean( np.mean((self.y_test - self.y_pred_matrix)**2, axis=1, keepdims=True) )
        bias = np.mean( (self.y_test - np.mean(self.y_pred_matrix, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(self.y_pred_matrix, axis=1, keepdims=True) )

        return mse, bias, variance


    def degree_range_simul(self):
        """ Runs bootstrap resampling and bias-variance decomposition for a range of polynomial degrees.

            Parameters:
            :: p_range (1D array) = range of polynomial degrees
        
            ------------------------------------------------------------------
            Returns:
            :: mse_array (1D array) = array of total errors for each polynomial degree
            :: bias_array (1D array) = array of biases for each polynomial degree
            :: variance_array (1D array) = array of variances for each polynomial degree
        """

        # loop over polynomial degrees
        for p in range(len(self.p_range)):
            self.bootstrap(p=p, regression_method='OLS')
            mse, bias, variance = self.decompose_mse()

            self.mse_array[p] = mse
            self.bias_array[p] = bias
            self.variance_array[p] = variance

        return self.mse_array, self.bias_array, self.variance_array


    def optimal_degree(self):
        """ Returns the optimal polynomial degree that minimizes the total error (MSE). """

        if self.mse_array is None:
            raise ValueError("The degree range simulation must be run before finding the optimal degree (MSE array is None)")

        return np.argmin(self.mse_array)


    def visualize(self):
        """ Visualizes the bias-variance tradeoff. """

        if self.y_pred_matrix is None:
            raise ValueError("The bootstrap must be run before visualizing (predicted y matrix is None)")
        
        # plot
        plt.figure(figsize=(14/2.54, 10/2.54))
        plt.plot(self.p_range, self.mse_array, label=r"Total Error (MSE)", color='slateblue', linewidth=3.5)
        plt.plot(self.p_range, self.bias_array, label='Bias²', color='indianred')
        plt.plot(self.p_range, self.variance_array, label='Variance', color='darkgreen')
        plt.axvline(x=self.optimal_degree(), ymin=0, ymax=np.max(self.mse_array), color='gray', linestyle='dashed', label=f"Optimal Degree = {self.optimal_degree()}")
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Error')
        plt.title(f"Bias-Variance tradeoff for OLS regression with {self.n_bootstraps} bootstraps, {len(self.y_test)} test points")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()



def k_fold_cv(x, y, k, p_range, regression_method='OLS'):
    """ K-Fold type resampling for linear regression and MSE analysis.

        Parameters:
        :: x (1D array) = input dataset
        :: y (1D array) = true value to be modeled
        :: p (int) = polynomial degree
        :: regression_method (str) = 'OLS' or 'Ridge'

        ------------------------------------------------------------------
        Returns:
        :: estimated_mse_array = MSED as a function of p

    """
    # error for wrong input
    if regression_method != 'OLS' and regression_method != 'Ridge':
        raise ValueError(f"'regression_method' must be either 'OLS' or 'Ridge' and not {regression_method}")

    # initialize the K-Folding object from sklearn
    kfold = KFold(n_splits = k)
    
    cv_mse_matrix = np.zeros((len(p_range), k))

    for i, p in enumerate(p_range):
        
        for j, (train_inds, test_inds) in enumerate(kfold.split(x)):
            x_train = x[train_inds]
            y_train = y[train_inds]
            x_test = x[test_inds]
            y_test = y[test_inds]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
            x_test = scaler.transform(x_test.reshape(-1, 1)).flatten()
            y_test = y_test.reshape(-1, 1)

            lr = LinearRegression_own(intercept=True)           # initialize the LinearRegression_own object
            X_train = lr.polynomial_features(x_train, p)        # transform train set to polynomial features
            X_test = lr.polynomial_features(x_test, p)          # transform test set to polynomial features
            beta = lr.fit(X_train, y_train, method=regression_method)              # fit model to the i-th bootstrap train sample
            y_pred = lr.predict(X_test, beta).ravel()          # predict on the same test set at each i-th iteration

            cv_mse_matrix[i, j] = np.sum((y_pred - y_test)**2) / np.size(y_pred)

    estimated_mse_array_KFold = np.mean(cv_mse_matrix, axis = 1)  


    return estimated_mse_array_KFold

#--------------------------------
# TEST PART: TESTING AND VISUALIZATION
#--------------------------------

# matrices for MSE and R2 as a function of p and n
def MSE_R2_pn():
    """ Evaluates MSE and R2 as a function of complexity `p` and dimensionality `n` for OLS regression

        ------------------------------
        Returns:
        :: MSE_train_matrix 
        :: MSE_test_matrix
        :: Rsquared_train_matrix
        :: Rsquared_test_matrix
        all of shape (n, p)
    """

    n_range = np.linspace(10, 100000, 10).astype(int)
    p_range = np.arange(1, 26, 1)

    MSE_train_matrix = np.zeros((len(n_range), len(p_range)))
    MSE_test_matrix = np.zeros((len(n_range), len(p_range)))
    Rsquared_train_matrix = np.zeros((len(n_range), len(p_range)))
    Rsquared_test_matrix = np.zeros((len(n_range), len(p_range)))

    for i, n in enumerate(n_range):
        
        xi = np.linspace(-1, 1, n)
        yi = Runge(xi, noise=False)
        xi_train, xi_test, yi_train, yi_test = train_test_split(xi, yi, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        xi_train = scaler.fit_transform(xi_train.reshape(-1, 1)).flatten()
        xi_test = scaler.transform(xi_test.reshape(-1, 1)).flatten()

        for j, pi in enumerate(p_range):
            # suggested by DeepSeek: skip if p > n_train
            n_train = len(xi_train)
            if pi >= n_train:  # p features vs n_train samples
                MSE_train_matrix[i, j] = np.nan
                MSE_test_matrix[i, j] = np.nan
                Rsquared_train_matrix[i, j] = np.nan
                Rsquared_test_matrix[i, j] = np.nan
                continue
            
            # perform regression
            lr = LinearRegression_own(intercept=True)
            Xi_train = lr.polynomial_features(xi_train, pi)
            Xi_test = lr.polynomial_features(xi_test, pi)

            beta = lr.fit(Xi_train, yi_train, method='OLS')

            yi_pred_train = lr.predict(Xi_train, beta)
            yi_pred_test = lr.predict(Xi_test, beta)
            
            # update estimators matrices
            MSE_train_matrix[i, j] = mean_squared_error(yi_train, yi_pred_train)
            MSE_test_matrix[i, j] = mean_squared_error(yi_test, yi_pred_test)
            Rsquared_train_matrix[i, j] = r2_score(yi_train, yi_pred_train)
            Rsquared_test_matrix[i, j] = r2_score(yi_test, yi_pred_test)
    
    return MSE_train_matrix, MSE_test_matrix, Rsquared_train_matrix, Rsquared_test_matrix