import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_classif
from scipy import stats
import statsmodels.api as sm
import pandas as pd

def SimulateBreak():

    '''
    Creates a simulated time-series where there are two linear trend seperated
    by a break.
    '''

    # Generate 100 random normal points for x
    x = np.arange(1, 101)
    # Generate random error terms with small standard deviation
    epsilon = np.random.normal(0, 0.25, 100)
    # Create corresponding y values for the first segment
    y1 = -1 + 0.5 * x + np.random.normal(0, 2, 100)
    # Create corresponding y values for the second segment
    y2 = 3 + 0.2 * x + np.random.normal(0, 2, 100)
    # Combine both segments
    y = np.concatenate([y1[:50], y2[50:]])
    return x,y

def SimulateLinear():

    '''
    Creates a simulated time-series with a positive linear trend
    '''

    # Generate 25 sequential points for x
    x = np.arange(1, 101)

    # Generate corresponding y values with a linear relationship to x
    # Let's assume the relationship is y = 2x + 3
    # We add more random noise to y to make the data noisier
    y = 2 * x + 3 + np.random.normal(0, 2, 100)  # Increased standard deviation for more noise
    return x,y


def SimulateQuadratic():

    '''
    Creates a simulated time-series with a quadratic relationship
    y = 2x^2 + 3x + 4.

    Args:

    Returns:
        Two 1D arrays (x,y)
        
    '''
    x = np.arange(1, 26)
    y = 2 * x**2 + 3 * x + 4 + np.random.normal(0, 20, 25) 

    return x,y    

def reshape(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y

def linear_regression(x, y):
    model = LinearRegression().fit(x, y)
    predicted_y = model.predict(x)
    error = y - predicted_y
    RSS = np.sum(np.square(error))
    return RSS, predicted_y, model, error


def linear_regression2(x, y):
    """
    Perform a linear regression using Ordinary Least Squares (OLS).

    Args:
        x (pd.DataFrame): The independent variable(s).
        y (pd.DataFrame): The dependent variable.

    Returns:
        tuple: RSS, predicted_y, model, error, statistical_test
    """
    model = LinearRegression().fit(x, y)
    predicted_y = model.predict(x)
    error = y - predicted_y
    RSS = np.sum(np.square(error))

    # Add an intercept for the statistical test
    x_with_intercept = sm.add_constant(x)
    results = sm.OLS(y, x_with_intercept).fit()
    A = np.identity(len(results.params))
    A = A[1:, :]
    statistical_test = results.f_test(A)

    return RSS, predicted_y, model, error, statistical_test

def quadratic_regression(x, y):
    """
    Perform quadratic regression using Ordinary Least Squares (OLS).

    Args:
        x (pd.DataFrame): The independent variable(s).
        y (pd.DataFrame): The dependent variable.

    Returns:
        tuple: RSS, predicted_y, model, error, statistical_test
    """
        
    # Fit a quadratic polynomial model (degree = 2)
    coeffs = np.polyfit(x, y, 2)
    model = np.poly1d(coeffs)

    # Calculate predicted values
    predicted_y = model(x)
    error = y - predicted_y
    RSS = np.sum(np.square(error))

    # Add an intercept for the statistical test
    x_with_intercept = sm.add_constant(x)
    results = sm.OLS(y, x_with_intercept).fit()
    A = np.identity(len(results.params))
    A = A[1:, :]
    statistical_test = results.f_test(A)

    return RSS, predicted_y, model, error, statistical_test

def calculate_BIC(RSS, n, k):

    """
    Calculate the Bayesian Information Criterion (BIC).

    Args:
        RSS (float): Residual sum of squares (sum of squared residuals).
        n (int): Number of observations (sample size).
        k (int): Number of parameters estimated by the model.

    Returns:
        float: The BIC value.
    """

    term1 = math.log(RSS / n)
    term2 = (k + 1) / n * math.log(n)
    BIC_value = term1 + term2
    return BIC_value

def anova_f_test(X, residuals):
    """
    Performs ANOVA F-test on features X and target y.

    Args:
        X (array-like): Feature matrix with shape (n_samples, n_features).
        y (array-like): Target vector with shape (n_samples,).

    Returns:
        F_statistic (array): F-statistic values for each feature.
        p_values (array): Associated p-values.
    """
    F_statistic, p_values = f_classif(X, residuals)
    return F_statistic, p_values


def confidence_interval(x, y, model):
    # Predict y values for given x
    predicted_y = model.predict(x)
    # Calculate the residuals
    residuals = y - predicted_y
    # Calculate the mean squared error
    mse = np.sum((residuals)**2) / (len(x) - 2)
    # Calculate the standard error
    se = np.sqrt(mse)
    # Calculate the confidence interval
    interval = 1.96 * se
    return predicted_y - interval, predicted_y + interval


def segment(x, y):

    '''
    Iteratively fit a split line regression on either side of a moving breakpoint. 
    For each iteration it will calculate BIC using the sum of RSS for both lines. Returns 
    the segments with the lowest BIC, which should be the optimal fit.
    
    '''

    best_BIC = float('inf')
    best_segment_1 = (x, y)  
    best_segment_2 = (x, y)
    x, y = reshape(x, y)
    
    for i in range(5, len(x) - 5):  
        RSS_1, predicted_y_1, model_1, error, f = linear_regression2(x[:i], y[:i])
        RSS_2, predicted_y_2, model_2, error, f = linear_regression2(x[i:], y[i:])
        RSS = RSS_1 + RSS_2
        BIC = calculate_BIC(RSS, len(x), 3)  

        if BIC < best_BIC:
            best_BIC = BIC
            best_segment_1 = (x[:i], predicted_y_1, model_1)
            best_segment_2 = (x[i:], predicted_y_2, model_2)

    return best_segment_1, best_segment_2, best_BIC


def find_overlap_float(x, y):

    '''
    Checks if the confidence intervals for two regression lines on either side of a 
    breakpoint overlap. If they do, it prints the range of the overlap. Used to check
    if the bilinear time-series is continuous or discontinuous.
    
    '''

    # Calculate the overlap range
    overlap_start = max(x[0], y[0])
    overlap_end = min(x[1], y[1])
    
    # Check if there's an overlap
    if overlap_start <= overlap_end:
        return (overlap_start, overlap_end)
    else:
        return None
    

def CompareBIC(x,y):

    '''
    This function will compare a segmented linear, linear, and quadric model fit to the same time
    series. The most appropriate fit is determined by the model which has the lowest BIC.

    '''
    segment1, segment2, lowest_BIC = segment(x, y)

    # Calculate BIC for linear model
    x, y = reshape(x, y)
    RSS_linear, predicted_y_linear, _ = linear_regression(x, y)
    BIC_linear = calculate_BIC(RSS_linear, len(x), 1)  # k = 1 for a linear model

    # Calculate BIC for quadratic model
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    RSS_quad, predicted_y_quad, _ = linear_regression(x_poly, y)
    BIC_quad = calculate_BIC(RSS_quad, len(x), 2)  # k = 2 for a quadratic model

    # 1 = linear regression, 2 = quadratic regression, 3 = segmented regression
    BIC_values = {1: BIC_linear, 2: BIC_quad, 3: lowest_BIC}
    min_BIC_model = min(BIC_values, key=BIC_values.get)

    return(min_BIC_model)


def calculate_weights_and_evidence_ratio(bic_values):
    # Ensure there are exactly three BIC values
    if len(bic_values) != 3:
        raise ValueError("There must be exactly three BIC values.")
    
    # Find the minimum BIC value
    bic_min = min(bic_values)
    
    # Calculate the differences Δi
    delta_i = [bic - bic_min for bic in bic_values]
    
    # Calculate the weights
    exp_neg_half_delta = np.exp(-0.5 * np.array(delta_i))
    weights = exp_neg_half_delta / np.sum(exp_neg_half_delta)
    
    # Assign weights to specific regression models
    w_segmented = weights[0]
    w_linear = weights[1]
    w_quadratic = weights[2]
    
    # Calculate the evidence ratio Bw
    Bw = w_segmented / max(w_linear, w_quadratic)
    
    return {
        'segmented_weight': w_segmented,
        'linear_weight': w_linear,
        'quadratic_weight': w_quadratic,
        'evidence_ratio': Bw
    }



    

