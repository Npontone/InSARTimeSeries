import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression


# Generate 100 random normal points for x
x = np.arange(1, 101)

# Generate random error terms with small standard deviation
epsilon = np.random.normal(0, 0.25, 100)

# Create corresponding y values for the first segment
y1 = -1 + 0.5 * x + np.random.normal(0, 0.25, 100)

# Create corresponding y values for the second segment
y2 = 3 + 0.2 * x + np.random.normal(0, 0.25, 100)

# Combine both segments
y = np.concatenate([y1[:50], y2[50:]])

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

segment1, segment2, lowest_BIC = segment(x, y)
x, y = reshape(x, y)

# Create a scatterplot of the original data
plt.scatter(x, y, label="Original Data", color="blue")

# Plot the regression lines for segment 1 and segment 2
segment_1_x, segment_1_predicted_y, model_1 = segment1
segment_2_x, segment_2_predicted_y, model_2 = segment2

plt.plot(segment_1_x, segment_1_predicted_y, label="Segment 1", color="red")
plt.plot(segment_2_x, segment_2_predicted_y, label="Segment 2", color="green")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Segmented Regression with Breakpoint (Minimum 5 Points per Segment)")
plt.legend()
plt.grid(True)
plt.show()