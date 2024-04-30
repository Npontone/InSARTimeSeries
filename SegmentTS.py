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
    best_segment_1 = (x, y)  # Initialize with entire dataset
    best_segment_2 = (x, y)
    x, y = reshape(x, y)
    
    for i in range(5, len(x) - 5):  # Ensure at least 5 points in each segment
        RSS_1, predicted_y_1, model_1 = linear_regression(x[:i], y[:i])
        RSS_2, predicted_y_2, model_2 = linear_regression(x[i:], y[i:])
        RSS = RSS_1 + RSS_2
        BIC = calculate_BIC(RSS, len(x), 3)  # k = 3 for a two-line regression

        if BIC < best_BIC:
            best_BIC = BIC
            best_segment_1 = (x[:i], predicted_y_1, model_1)
            best_segment_2 = (x[i:], predicted_y_2, model_2)

    return best_segment_1, best_segment_2, best_BIC


x,y = SimulateLinear()
segment1, segment2, lowest_BIC = segment(x, y)
x, y = reshape(x, y)

# Calculate BIC for linear model
RSS_linear, predicted_y_linear, _ = linear_regression(x, y)
BIC_linear = calculate_BIC(RSS_linear, len(x), 1)  # k = 1 for a linear model

# Calculate BIC for quadratic model
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
RSS_quad, predicted_y_quad, _ = linear_regression(x_poly, y)
BIC_quad = calculate_BIC(RSS_quad, len(x), 2)  # k = 2 for a quadratic model

# Compare BICs
print(f"BIC for Two Line model: {lowest_BIC}")
print(f"BIC for linear model: {BIC_linear}")
print(f"BIC for quadratic model: {BIC_quad}")

BIC_values = {"Linear Model": BIC_linear, "Quadratic Model": BIC_quad, "Segmented Regression": lowest_BIC}
min_BIC_model = min(BIC_values, key=BIC_values.get)
print(f"The model with the lowest BIC is the {min_BIC_model}.")

# Create a scatterplot of the original data
plt.scatter(x, y, label="Original Data", color="blue")

# Plot the regression lines for segment 1 and segment 2
segment_1_x, segment_1_predicted_y, model_1 = segment1
segment_2_x, segment_2_predicted_y, model_2 = segment2

# Calculate the confidence intervals for both segments
lower_1, upper_1 = confidence_interval(segment_1_x, y[:len(segment_1_x)], model_1)
lower_2, upper_2 = confidence_interval(segment_2_x, y[len(segment_1_x):], model_2)


def find_overlap_float(x, y):
    # Calculate the overlap range
    overlap_start = max(x[0], y[0])
    overlap_end = min(x[1], y[1])
    
    # Check if there's an actual overlap
    if overlap_start <= overlap_end:
        return (overlap_start, overlap_end)
    else:
        return None
    
set1 = (lower_1[-1], upper_1[-1])  # First set of points (vertical positions)
set2 = (lower_2[0], upper_2[0])  # Second set of points (vertical positions)

overlap_range = find_overlap_float(set1, set2)

if overlap_range:
    print("95% CI Overlap")
else:
    print("No 95% CI overlap")

# Check if the confidence intervals overlap
#overlap = np.max([np.max(lower_1), np.max(lower_2)]) < np.min([np.min(upper_1), np.min(upper_2)])
#print(f"Do the confidence intervals overlap? {overlap}")

plt.plot(segment_1_x, segment_1_predicted_y, label="Segment 1", color="red")
plt.plot(segment_2_x, segment_2_predicted_y, label="Segment 2", color="green")

# Plot the regression line for the linear model
plt.plot(x, predicted_y_linear, label="Linear Model", color="purple")

# Plot the regression line for the quadratic model
plt.plot(x, predicted_y_quad, label="Quadratic Model", color="orange")

# Add the confidence intervals to the plot
plt.fill_between(segment_1_x.flatten(), lower_1.flatten(), upper_1.flatten(), color='red', alpha=0.1, label="Confidence Interval 1")
plt.fill_between(segment_2_x.flatten(), lower_2.flatten(), upper_2.flatten(), color='green', alpha=0.1, label="Confidence Interval 2")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Segmented Regression with Breakpoint (Minimum 5 Points per Segment)")
plt.legend()
plt.grid(True)
plt.show()




