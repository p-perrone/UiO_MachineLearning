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

This module contains utility functions for Neural Network implementation, 
including:
- Activation functions and their derivatives
- Cost functions and their derivatives
- Data preprocessing and scaling utilities
- Evaluation metrics such as accuracy and confusion matrix
- Any additional helper functions for training, prediction, and regularization
"""

# --------------------------------
# dependencies
import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from autograd import grad, elementwise_grad
import copy
import time
from typing import Literal, List, Optional

# own functions
from ml_project1.functions import *

#------------------------------------------------#
# Activation functions
#------------------------------------------------#

def ReLU(z):
    """ Computes Rectified Linear Unit:
        f(z) = z if z > 0, 0 otherwise
    """
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    """ Derivative of ReLU:
        f'(z) = 1 if z > 0, 0 otherwise
    """
    return np.where(z > 0, 1, 0)

#------------------------------------------------#

def leaky_ReLU(z):
    """ Computes Leaky ReLU:
        f(z) = z if z > 0, alpha*z otherwise
    """
    return np.where(z > 0, z, 0.01 * z)

def leaky_ReLU_der(z):
    """ Derivative of Leaky ReLU:
        f'(z) = 1 if z > 0, alpha otherwise
    """
    return np.where(z > 0, 1, 0.01)

#------------------------------------------------#

def ELU(z):
    """ Computes Exponential Linear Unit (ELU):
        f(z) = z if z > 0, alpha*(exp(z)-1) otherwise
    """
    return np.where(z > 0, z, np.exp(z) - 1)

def ELU_der(z):
    """ Derivative of ELU:
        f'(z) = 1 if z > 0, f(z)+alpha otherwise
    """
    return np.where(z > 0, 1, np.exp(z))

#------------------------------------------------#

def sigmoid(z):
    """ Computes Sigmoid activation:
        f(z) = 1 / (1 + exp(-z))
    """
    z = np.clip(z, -500, 500)  # avoids overflow in exp
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    """ Derivative of Sigmoid:
        f'(z) = f(z) * (1 - f(z))
    """
    s = sigmoid(z)
    return s * (1 - s)

#------------------------------------------------#

def tanh(z):
    """ Computes hyperbolic tangent activation:
        f(z) = tanh(z)
    """
    return np.tanh(z)

def tanh_der(z):
    """ Derivative of tanh:
        f'(z) = 1 - tanh(z)^2
    """
    return 1 - np.tanh(z)**2

#------------------------------------------------#
def softmax(z):
    """ Computes softmax activation:
            f(z_i) = exp(z_i) / sum_j exp(z_j)
    """
    z = z - np.max(z, axis=-1, keepdims=True)  # stability
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-10)

def softmax_der(z):
    """ Element-wise approximation derivative of softmax (not full Jacobian)
        Useful for cross-entropy backprop where simplifications exist.
    """
    s = softmax(z)
    return s * (1 - s)

#------------------------------------------------#
# Cost functions
#------------------------------------------------#

def mse(predict, targets, weights=None, regression_method='ols', lbda=0.0, dtype=None):  # DeepSeek: add dtype parameter
    """
    Mean Squared Error cost function
    """
    residuals = predict - targets
    
    if regression_method == 'ols':
        return np.mean(residuals**2)
    elif regression_method == 'ridge':
        return np.mean(residuals**2) + lbda * np.sum(weights**2)
    elif regression_method == 'lasso':
        return np.mean(residuals**2) + lbda * np.sum(np.abs(weights))
    

def mse_der(predict, targets, autodiff : bool= False):
    """ Computes gradient of MSE w.r.t predictions """
    if autodiff:
        return elementwise_grad(mse)(predict, targets)

    n = targets.shape[0]
    residuals = predict - targets
    grad_pred = (2 / n) * residuals
    return grad_pred

#------------------------------------------------#

def cross_entropy(predict, targets):
    """ Computes Cross-Entropy loss for multi-class classification """
    n = targets.shape[0]
    return -np.sum(targets * np.log(predict + 1e-10)) / n

def cross_entropy_der(predict, targets, autodiff : bool= False):
    """ Computes gradient of Cross-Entropy w.r.t predictions """
    if autodiff:
        return elementwise_grad(cross_entropy)(predict, targets)
    else:
        return (predict - targets) / predict.shape[0]
    

#------------------------------------------------#
# dictionary containing all the previous functions:
#------------------------------------------------#

activation_functions_dic = {
    'sigmoid'    : (sigmoid, sigmoid_der),
    'tanh'       : (tanh, tanh_der),
    'softmax'    : (softmax, softmax_der),
    'relu'       : (ReLU, ReLU_der),
    'leaky_relu' : (leaky_ReLU, leaky_ReLU_der),
    'elu'        : (ELU, ELU_der),
    }

cost_functions_dic = {
    'mse'           : (mse, mse_der),
    'cross_entropy' : (cross_entropy, cross_entropy_der)
    }

#------------------------------------------------#
# function for setting activation:
#------------------------------------------------#

def set_activations(
        functions_dic: dict,
        functions_names_list: list,
        automatic_diff: bool = False):
    """
    Selects activation functions and their derivatives from a dictionary 
    and returns two lists: one with the functions and one with the derivatives.

    Parameters
    ----------
    :: functions_dic : dict
        Dictionary mapping function names to a tuple/list: (function, manual_derivative)
    :: functions_names_list : list
        List of names of the activation functions to use in each layer
    :: automatic_diff : bool, default=False
        If True, uses automatic differentiation to compute derivatives; 
        otherwise uses the provided manual derivatives

    Returns
    -------
    :: activation_funcs : list
        List of activation functions corresponding to `functions_names_list`
    :: activation_funcs_ders : list
        List of derivatives (manual or automatic) corresponding to the activations
    """
    
    activation_funcs = list()
    activation_funcs_ders = list()

    for func_name in functions_names_list:
        for key in functions_dic.keys():
            if func_name == key:
                # get f(z) and f'(z) from the dictionary
                func = functions_dic[key][0]
                manual_der = functions_dic[key][1]
                
                # automatic derivative
                auto_der = elementwise_grad(func)

                activation_funcs.append(func)
             
                if automatic_diff:
                    activation_funcs_ders.append(auto_der)
                else:
                    activation_funcs_ders.append(manual_der)

    return activation_funcs, activation_funcs_ders


#------------------------------------------------#
# handling scheduler out of the NN class:
#------------------------------------------------#

class Scheduler:
    """
    abstract class for schedulers
    """

    def __init__(self, eta=1e-3):
        self.eta = eta  # base learning rate

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient  # simple constant step

    def reset(self):
        pass  # nothing to reset


class RMSProp(Scheduler):
    def __init__(self, eta, beta=0.99):
        super().__init__(eta)
        self.beta = beta  # smoothing factor
        self.second = 0.0  # moving average of squared gradients

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero
        self.second = self.beta * self.second + (1 - self.beta) * gradient**2  # update avg
        return self.eta * gradient / (np.sqrt(self.second + delta))  # rescale step

    def reset(self):
        self.second = 0.0  # reset moving average


class Adam(Scheduler):
    def __init__(self, eta, beta1=0.9, beta2=0.999):
        super().__init__(eta)
        self.beta1 = beta1  # momentum factor
        self.beta2 = beta2  # rms factor
        self.moment = 0  # first moment
        self.second = 0  # second moment
        self.n_epochs = 1  # counter for bias correction

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        # update biased first and second moments
        self.moment = self.beta1 * self.moment + (1 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1 - self.beta2) * gradient**2

        # compute bias-corrected estimates
        moment_corrected = self.moment / (1 - self.beta1**self.n_epochs)
        second_corrected = self.second / (1 - self.beta2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))  # scaled step

    def reset(self):
        self.n_epochs += 1  # increment epoch
        self.moment = 0  # reset first moment
        self.second = 0  # reset second moment


class NeuralNetwork:
    """
    Creates Neural Network object, adapted to batch inputs of shape \
    (< number of samples >, < number of features >). 
    It supports:
    - layers creation
    - simple Feed Forward algorithm
    - cost function definition
    - Feedforward with variables saving
    - Backpropagation
    - training
    - layers reset

    """

    def __init__(
        self,
        network_input_size,
        layer_output_sizes : List[int],
        activation_types : List[Literal['relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh']] = None,
        autodiff : bool = False,
        seed=42,
        cost_type : Literal['mse', 'cross_entropy'] = "mse",
        regression_method : Literal['ols', 'ridge', 'lasso'] = "ols",
        lbda1=0,
        lbda2=0,
        verbose : bool=False
    ):
        """
        Initializes Neural Network with layers, activation functions, cost type, and regularization.

        Parameters
        ----------
        :: network_input_size : int
            Number of input features
        :: layer_output_sizes : list of int
            Number of neurons per layer
        :: activation_types : list of str
            Activation functions per layer
        :: autodiff : bool, default=False
            If True, compute activation derivatives automatically
        :: seed : int, default=42
            Random seed for weight initialization
        :: cost_type : str, default='mse'
            Cost function: 'mse' or 'cross_entropy'
        :: regression_method : str, default='ols'
            Regression type: 'ols', 'ridge', or 'lasso'
        :: lbda1 : float, default=0
            L1 regularization penalty
        :: lbda2 : float, default=0
            L2 regularization penalty
        :: verbose : bool, default=False
            If True, prints progress during training
        """

        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_types = activation_types
        self.autodiff = autodiff
        self.activation_funcs, self.activation_ders = set_activations(
            activation_functions_dic, 
            self.activation_types,
            self.autodiff
            )
        self.seed = seed
        self.cost_type = cost_type
        self.regression_method = regression_method
        self.lbda1 = lbda1
        self.lbda2 = lbda2

        self.weights_biases = list()
        self.weights_biases_grads = list()
        self.layer_inputs = list()
        self.Zs = list()
        self.scheduler = None
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.verbose = verbose


    def _create_layers(self):
        """ 
        Initializes weights and biases for all layers randomly.

        Returns
        -------
        :: weights_biases : list of tuples
            Each tuple contains (W, b) for a layer
        """

        # ChatGPT: clear before creating new (W,b)
        self.weights_biases.clear()

        i_size = self.network_input_size

        for layer_output_size in self.layer_output_sizes:
            np.random.seed(self.seed)

            W = np.random.randn(i_size, layer_output_size)
            b = np.zeros((1, layer_output_size))
            self.weights_biases.append((W, b))

            i_size = layer_output_size

        self.weights_biases_grads = [() for _ in self.weights_biases]    

        return self.weights_biases


    def cost(self, predict, targets):
        """
        Computes cost given predictions and targets.

        Parameters
        ----------
        :: predict : array
            Predicted outputs
        :: targets : array
            True outputs

        Returns
        -------
        :: cost_value : float
            Value of the cost function
        """
        if self.cost_type == "mse":
            return mse(predict, targets)
        elif self.cost_type == "cross_entropy":
            return cross_entropy(predict, targets)
        else:
            raise ValueError(f"Cost type '{self.cost_type}' not defined. Can be 'mse' or 'cross_entropy'.")
        
    
    def cost_der(self, predict, targets):
        """
        Computes gradient of cost function w.r.t predictions.

        Parameters
        ----------
        :: predict : array
            Predicted outputs
        :: targets : array
            True outputs

        Returns
        -------
        :: grad : array
            Gradient of cost w.r.t predictions
        """

        if self.cost_type == "mse":
            return mse_der(predict, targets, autodiff=self.autodiff)
        elif self.cost_type == "cross_entropy":
            return cross_entropy_der(predict, targets, self.autodiff)
        else:
            raise ValueError(f"Cost type '{self.cost_type}' not defined. Can be 'mse' or 'cross_entropy'.")


    def _feedforward(self, inputs):
        """
        Performs feedforward pass and stores activations and pre-activations.

        Parameters
        ----------
        :: inputs : array
            Input data

        Returns
        -------
        :: layer_inputs : list
            Activations from each layer before current layer
        :: Zs : list
            Pre-activation values (W*A + b)
        :: A : array
            Final output of the network
        """

        if self.weights_biases is None:
            raise ValueError("`self.layers` is still None (empty tuple). You must call `_create_layers()` "\
                             "before performing Feed Forward or Back Propagation.")

        if np.shape(inputs)[1] == self.network_input_size: 
            A = inputs
        else:
            raise ValueError(f"Second dimension of inputs must match `network_input_size` (= number of features). "\
                             f"Inputs matrix has shape {inputs.shape}, but it should be {inputs.shape[0], self.network_input_size}. \n"\
                             "Either change your inputs matrix or `network_input_size`")
        
        self.layer_inputs = []
        self.Zs = []

        for (W, b), activation_func in zip(self.weights_biases, self.activation_funcs):
            
            self.layer_inputs.append(A)
            Z = A @ W + b
            A = activation_func(Z)
            self.Zs.append(Z)

        return self.layer_inputs, self.Zs, A
    

    def _backpropagate(self, inputs, targets):
        """
        Performs backpropagation, computing gradients for all weights and biases.

        Parameters
        ----------
        :: inputs : array
            Input data
        :: targets : array
            True output labels

        Returns
        -------
        :: weights_biases_grads : list of tuples
            Gradients for each layer (dW, db)
        """
        
        if self.weights_biases is not None:
            layer_inputs, zs, predict = self._feedforward(inputs)
        else:
            raise ValueError("`self.layers` is still None (empty tuple). You must call `_create_layers()` "\
                             "before performing Feed Forward or Back Propagation.")

        for i in reversed(range(len(self.weights_biases))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.activation_ders[i]

            # DeepSeek fixed backprop for softmax + cross-entropy
            if i == len(self.weights_biases) - 1:
                if self.activation_types[-1] == 'softmax' and self.cost_type == 'cross_entropy':
                    delta = self.cost_der(predict, targets)  # No activation_der needed!
                else:
                    dC_da = self.cost_der(predict, targets)
                    delta = dC_da * activation_der(z)
            else:
                W_next, _ = self.weights_biases[i + 1]
                dC_da = delta @ W_next.T
                delta = dC_da * activation_der(z) 

            grad_weights = layer_input.T @ delta
            grad_biases = np.sum(delta, axis=0, keepdims=True) / inputs.shape[0]

            if self.lbda1 != 0:
                grad_weights += (self.lbda1 / inputs.shape[0]) * np.sign(self.weights_biases[i][0])
            if self.lbda2 != 0:
                grad_weights += (2 * self.lbda2 / inputs.shape[0]) * self.weights_biases[i][0]

            self.weights_biases_grads[i] = (grad_weights, grad_biases)

        return self.weights_biases_grads
    

    def _define_scheduler(self, 
                          type="constant", 
                          eta=1e-3, 
                          beta=0.99, 
                          beta1=0.9, 
                          beta2=0.999):
        """
        Defines the optimizer (scheduler) for gradient updates.

        Parameters
        ----------
        :: type : str, default='constant'
            Type of scheduler: 'constant', 'rmsprop', 'adam'
        :: eta : float, default=1e-3
            Learning rate
        :: beta : float, default=0.99
            RMSProp smoothing factor
        :: beta1 : float, default=0.9
            Adam momentum factor
        :: beta2 : float, default=0.999
            Adam RMS factor

        Returns
        -------
        :: scheduler : Scheduler object
            Initialized scheduler
        """
        
        if type == "constant":
            self.scheduler = Constant(eta)
        elif type == "rmsprop":
            self.scheduler = RMSProp(eta, beta=beta)
        elif type == "adam":
            self.scheduler = Adam(eta, beta1=beta1, beta2=beta2)
        else:
            raise ValueError(f"Scheduler type '{type}' not defined. Can be 'constant', 'rmsprop' or 'adam'.")
        
        # this is a ChatGPT fix
        self.schedulers_weight = [copy.deepcopy(self.scheduler) for _ in self.weights_biases]
        self.schedulers_bias = [copy.deepcopy(self.scheduler) for _ in self.weights_biases]
        
        return self.scheduler


    def _update_weights(self):
        """
        Updates weights and biases of all layers using gradients and scheduler.
        """

        if len(self.weights_biases) == 0:
            raise ValueError(f"`self.weights_biases` is None. You must call `_create_layers()` before updating the weigths.") 
        
        if self.scheduler == None:
            raise ValueError(f"`self.scheduler` is None. You must call `_define_scheduler()` before updating the weigths.")
        
        if self.weights_biases_grads == None:
            raise ValueError(f"`self.weigths_biases_grad` is None. You must call `_backpropagate()` before updating the weigths.")
        
        for i, (W, b) in enumerate(self.weights_biases):
                dW, db = self.weights_biases_grads[i]
                W -= self.schedulers_weight[i].update_change(dW)
                b -= self.schedulers_bias[i].update_change(db)
                self.weights_biases[i] = (W, b)

    
    def _train(self,
              inputs,
              targets,
              n_epochs:  int=250,
              n_batches: int=20,
              ):
        """
        Trains the neural network with mini-batch Stochastic Gradient Descent.

        Parameters
        ----------
        :: inputs : array
            Input data
        :: targets : array
            True outputs
        :: n_epochs : int, default=250
            Number of training epochs
        :: n_batches : int, default=20
            Number of mini-batches per epoch
        """
        
        n_samples = inputs.shape[0]
        batch_size = n_samples // n_batches

        for epoch in range(n_epochs):
            # this indexing shuffle is a ChatGPT hint that helps avoiding dimension mismatches
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            inputs, targets = inputs[indices], targets[indices]

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                inputs_batch = inputs[start:end]
                targets_batch = targets[start:end]
                    
                self._backpropagate(inputs_batch, targets_batch)
                self._update_weights()

            # DeepSeek hint
            if (epoch + 1) % 50 == 0:
                preds = self._predict(inputs)
                acc = self._accuracy(preds, targets)

                if self.verbose:
                    print(f"Epoch {epoch+1}/{n_epochs} - Accuracy: {acc:.4f}")

    
    def _predict(self, inputs): 
        """
        Computes network predictions for given inputs.

        Parameters
        ----------
        :: inputs : array
            Input data

        Returns
        -------
        :: predictions : array
            Network output
        """

        if self.weights_biases is None:
            raise ValueError("`self.layers` is still None (empty tuple). You must run `_create_layers()` "\
                             "before performing Feed Forward or Back Propagation.")

        if np.shape(inputs)[1] == self.network_input_size: 
            A = inputs
        else:
            raise ValueError(f"Second dimension of inputs must match `network_input_size` (= number of features). "\
                             f"Inputs matrix has shape {inputs.shape}, but it should be {inputs.shape[0], self.network_input_size}. \n"\
                             "Either change your inputs matrix or `network_input_size`")
        
        for (W, b), activation_func in zip(self.weights_biases, self.activation_funcs):
            Z = A @ W + b
            A = activation_func(Z)

        return A


    def _accuracy(self, predicts, targets):
        """
        Computes classification accuracy for predictions.

        Parameters
        ----------
        :: predicts : array
            Predicted outputs (probabilities or logits)
        :: targets : array
            True labels (one-hot or integers)

        Returns
        -------
        :: acc : float
            Classification accuracy
        """
        # predicted class indices
        pred_labels = np.argmax(predicts, axis=1)

        # handle one-hot targets
        if targets.ndim > 1:
            true_labels = np.argmax(targets, axis=1)
        else:
            true_labels = targets

        return np.mean(pred_labels == true_labels)


    def _reset_weights(self):
        """
        Resets all network weights, biases, and schedulers.
        """
        self.weights_biases.clear()
        self.weights_biases_grads.clear()
        self.layer_inputs.clear()
        self.Zs.clear()
        self.schedulers_weight.clear()
        self.schedulers_bias.clear()
        self._create_layers()

        if self.scheduler is not None:
            self.schedulers_weight = [copy.deepcopy(self.scheduler) for _ in self.weights_biases]
            self.schedulers_bias = [copy.deepcopy(self.scheduler) for _ in self.weights_biases]
