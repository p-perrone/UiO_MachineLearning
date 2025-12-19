# python3
# Source code with all the functions for "Applied Data Analysis and Machine Learning" master course
# University of Oslo, Autumn 2025
#---------------------------------
# author: Pietro Perrone
#
""" 
=====================
ml_project3.functions
=====================

This module contains utility functions for feature selection and Long-Short-Term Memory preprocessing implementation, 
including:
- feature selection with bootstrapping, Ridge and Lasso regressions
- Long Short-Term Memoy model implementation and preprocessing
"""

#--------------------------------
# dependencies
#------- general imports -------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

#------- multiprocessing ------
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
from numba import jit

#------- ML imports -------
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from sklearn.utils import resample


# own functions
from ml_project1.functions import *

#------------------------------------------------#
# selecting features with bootsrtrapped Ridge and/or Lasso fits,
# evluating the results
#------------------------------------------------#

def feature_selection(
        X, y,
        model_type="lasso",      # "lasso" or "ridge"
        alpha=1.0,
        n_boot=200,
        threshold_factor=0.05           # keep features with |coef| ≥ threshold * max|coef|
    ):
    """ Performs bootsrap to estimate feature coefficients using Ridge or Lasso regression; selects features to keep.
    ______________
    Parameters
        :: X : DataFrame | ndarray
            Training features (scaled).
        :: y : Series | ndarray
            Training target.
        :: model_type : str
            'ridge' or 'lasso'
        :: alpha : float
            Regularization strength.
        :: n_boot : int
            Number of bootstrap samples.
        :: threshold_factor : float
            Relative magnitude threshold for feature selection.
    ______________
    Returns:
        :: result : pd.DataFrame
            DataFrame with coefficients, std errors, CIs, and keep flag.
    """
    
    # linear model
    if model_type.lower() == "lasso":
        model = Lasso(alpha=alpha)
    elif model_type.lower() == "ridge":
        model = Ridge(alpha=alpha)
    else:
        raise ValueError("model_type must be 'lasso' or 'ridge'")

    model.fit(X, y)
    base_coefs = np.array(model.coef_)
    features = X.columns

    # bootstrap
    boot_coefs = np.zeros((n_boot, len(features)))

    for i in range(n_boot):
        Xb, yb = resample(X, y)
        m = Lasso(alpha=alpha) if model_type == "lasso" else Ridge(alpha=alpha)
        m.fit(Xb, yb)
        boot_coefs[i] = m.coef_

    # standard error of coefficients
    se = boot_coefs.std(axis=0)

    # 95% bootstrap percentile CI
    ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    ci_high = np.percentile(boot_coefs, 97.5, axis=0)

    # feature selection by relative magnitude
    max_coef = np.max(np.abs(base_coefs))
    threshold = threshold_factor * max_coef 
    keep = np.abs(base_coefs) >= threshold

    result = pd.DataFrame({
        "coef" : base_coefs,
        "std_error" : se,
        "ci_2.5%" : ci_low,
        "ci_97.5%" : ci_high,
        "keep" : keep
    }, index=features)

    return boot_coefs, result, threshold


def stability_mse(X, y, model_type='lasso', alphas=None, n_boot=200):
    """
    Fits Ridge or Lasso models across a range of alphas, computing sample-specific MSE and bootstrap sign-stability.
    ______________
    Parameters
        :: X : DataFrame | ndarray
            Training features (scaled).
        :: y : Series | ndarray
            Training target.
        :: model_type : str
            'ridge' or 'lasso'.
        :: alphas : list | ndarray
            Sequence of alpha values to evaluate. If None, uses logspace(-4, 0, 15).
        :: n_boot : int
            Number of bootstrap samples used to estimate sign stability.
    ______________
    Returns
        :: results : pd.DataFrame
            Table with columns:
                - 'alpha' : tested regularization strengths
                - 'mse' : sample-specific mean squared error
                - 'n_nonzero' : count of non-zero coefficients (Lasso only)
        :: stability : pd.DataFrame
            DataFrame indexed by alpha with one column per feature.
            Each entry is the fraction of bootstrap models whose coefficient sign
            matches the sign of the model performed on the full dataset at the same alpha.
    """
    if alphas is None:
        alphas = np.logspace(-4, 0, 4)

    if model_type == 'lasso':
        Model = lambda a: Lasso(alpha=a, max_iter=10000)
    elif model_type == 'ridge':
        Model = lambda a: Ridge(alpha=a)
    else:
        raise ValueError("model_type must be 'lasso' or 'ridge'.")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    results = []
    stability = {}

    for a in alphas:
        # fitting the model for the full dataset
        model = Model(a)
        model.fit(X_arr, y_arr)
        mse = mean_squared_error(y_arr, model.predict(X_arr))
        coefs = model.coef_

        # bootstrap model fitting 
        signs = []
        for _ in range(n_boot):
            idx = np.random.choice(len(y_arr), len(y_arr), replace=True)
            Xb, yb = X_arr[idx], y_arr[idx]
            mb = Model(a)
            mb.fit(Xb, yb)
            signs.append(np.sign(mb.coef_))
        
        # compute sign stability
        signs = np.vstack(signs)
        stability_a = (np.mean(signs == np.sign(coefs), axis=0))  # fraction of consistent sign
        stability[a] = stability_a

        results.append({
            "alpha": a,
            "mse": mse,
            "n_nonzero": np.sum(np.abs(coefs) > 1e-6)
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(stability, index=X.columns).T   # rows = alpha, cols = features
    )


#------------------------------------------------#
# LSTM preprocessing
#------------------------------------------------#


def create_sequence(time_series: np.ndarray | list | pd.DataFrame,
                    sequence_length: int,
                    forecast_horizon: int,
                    feature_cols: list = None,
                    target_col: int=None
                    ):
    """Creates sequences from a time series.
    ______________
    Params:
        :: time_series = full dataset containing past dayes measures and dayes to forecast. Daily or hourly step.
        :: sequence_length = length of each sequence 
        :: forecast_horizon = amount of forecasted time
        :: target_col = variable to be sequenced, if input is pd.DataFrame
    ______________
    Returns:
        :: X = sequenced array of shape used for training/testing (number of samples, sequence length) 
        :: Y = sequence of measures to be forecsted ("true" value)
    """
    # handling input type
    if isinstance(time_series, list):
        data = np.array(time_series)

    elif isinstance(time_series, pd.DataFrame):
        if target_col is None:
            raise ValueError("Specify target_col when passing a DataFrame.")
        if target_col not in time_series.columns:
            raise ValueError(f"'{target_col}' not found in DataFrame.")
        
        X_data = time_series[feature_cols].values
        Y_data = time_series[target_col].values.reshape(-1, 1)

        # stack features and target; 
        # last column is the target, which simplifies slicing later
        data = np.hstack([X_data, Y_data])

    else:
        data = np.array(time_series)


    # ensuring 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = len(data)

    X = []
    Y = []

    # last column = target
    for i in range(n - sequence_length - forecast_horizon):
        # input: full feature set for past sequence_length timesteps
        X.append(data[i : i + sequence_length, :-1])

        # output: future target only (not full vector)
        Y.append(data[i + sequence_length : i + sequence_length + forecast_horizon, -1])

    X = np.array(X)
    Y = np.array(Y)

    print(f"X shape = {X.shape}")
    print(f"Y shape = {Y.shape}")

    return X, Y


def make_bias_initializer(y_train):
    """Initialize bias for TensorFlow Dense layers.
    ______________
    Params:
        :: y_train : the true training set from which averaging values for the bias
    ______________
    Returns:
        :: bias_init : the bias initialization
    """
    last_vals = y_train[-12:].flatten()
    bias_init = Constant(float(np.mean(last_vals)))

    return bias_init