from setuptools import setup, find_packages

setup(
    name="ml_project1",
    version="0.1.0",
    py_modules=["functions"],
    description="ML Project 1",
)


class LinearRegression_own:
    """
    Own implementation of linear regression with OLS and Ridge methods and their analytical solutions.
    Includes polynomial feature expansion and fitting.
    
    Based on scikit-learn linear regression workflow.
    """

    def __init__(self, intercept=True):
        self.intercept = intercept
        self.beta = None


    def polynomial_features(self, x, p):  
        """ Computes the design matrix for linear regression.
        
            Parameters:
            :: x (array) = input dataset
            :: p (int) = polynomial degree
            :: intercept (bool) = sets first column at 1 if True, at 0 if False
        """
        if not isinstance(self.intercept, bool):
            raise TypeError(f"Intercept must be boolean (True or False), not {type(self.intercept)}")
            
        n = len(x)
        
        if self.intercept:
            X = np.ones((n, p + 1))
            for j in range(1, p + 1):
                X[:, j] = x**(j)
        else:
            X = np.ones((n, p))
            for j in range(p):
                X[:, j] = x**(j + 1)

        return X


    def fit(self, X, y, method='OLS', lbda=0.1):
        """Fit linear regression model.
        
            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: y (array) = true value to be modeled
            :: method (str) = 'OLS' or 'Ridge'
            :: lbda (scalar) = penalty coefficient for Ridge regression
        """

        if method == 'OLS':
            self.beta = self.OLS_params(X, y)
        elif method == 'Ridge':
            self.beta = self.Ridge_params(X, y, lbda)
        else:
            raise ValueError(f"`method` must be 'OLS' or 'Ridge', not {method}")
        
        return self


    def OLS_params(self,X, y):
        """ Computes optimal parameters for ordinary least squares regression.

            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: y (array) = true value to be modeled
        """
        return np.linalg.pinv(X.T @ X) @ X.T @ y


    def Ridge_params(self, X, y, lbda):
        """ Computes optimal parameters for Ridge regression.

            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: y (array) = true value to be modeled
            :: lmbda (scalar) = penalty coefficient
        """
        n_features = X.shape[1]
        return np.linalg.inv(X.T @ X + lbda * np.eye(n_features)) @ X.T @ y


    def predict(self, X):
        """ Computes the regression curve (predicted y values).

            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept).fit(X, y, method, lbda)
        """
        if self.beta is None:
            raise ValueError("A model must be fitted before predicting")
        
        return X @ self.beta