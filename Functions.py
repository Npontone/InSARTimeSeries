import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

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
    return RSS, predicted_y, model

def calculate_BIC(RSS, n, k):
    term1 = math.log(RSS / n)
    term2 = (k + 1) / n * math.log(n)
    BIC_value = term1 + term2
    return BIC_value

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

    best_BIC = float('inf')
    best_segment_1 = (x, y)  
    best_segment_2 = (x, y)
    x, y = reshape(x, y)
    
    for i in range(5, len(x) - 5):  
        RSS_1, predicted_y_1, model_1 = linear_regression(x[:i], y[:i])
        RSS_2, predicted_y_2, model_2 = linear_regression(x[i:], y[i:])
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
    

    



