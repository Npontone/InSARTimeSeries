import Functions
import matplotlib.pyplot as plt


x,y = Functions.SimulateBreak()
segment1, segment2, lowest_BIC = Functions.segment(x, y)

x, y = Functions.reshape(x, y)

# Calculate BIC for linear model
RSS_linear, predicted_y_linear, linear_residuals, error_linear, f_linear = Functions.linear_regression2(x, y)
BIC_linear = Functions.calculate_BIC(RSS_linear, len(x), 1)  # k = 1 for a linear model

# Calculate BIC for quadratic model

poly = Functions.PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
RSS_quad, predicted_y_quad, quad_residuals, error_quad, f_quad = Functions.linear_regression2(x_poly, y)
BIC_quad = Functions.calculate_BIC(RSS_quad, len(x), 2)  # k = 2 for a quadratic model

# Compare BICs
print(f"BIC for Two Line model: {lowest_BIC}")
print(f"BIC for linear model: {BIC_linear}")
print(f"BIC for quadratic model: {BIC_quad}")

bic_values = [lowest_BIC,BIC_linear,BIC_quad]

BIC_values = {"Linear Model": BIC_linear, "Quadratic Model": BIC_quad, "Segmented Regression": lowest_BIC}
min_BIC_model = min(BIC_values, key=BIC_values.get)
print(f"The model with the lowest BIC is the {min_BIC_model}.")

# Calculate weights and evidence ratio
results = Functions.calculate_weights_and_evidence_ratio(bic_values)

bw = results['evidence_ratio']
bth = 1

if bw >= bth:
    print('Significant Breakpoint')

elif bw < bth:
    print('No significant Breakpoint')


print("Segmented Regression Weight:", results['segmented_weight'])
print("Linear Regression Weight:", results['linear_weight'])
print("Quadratic Regression Weight:", results['quadratic_weight'])
print("Evidence Ratio (Bw):", results['evidence_ratio'])

# Create a scatterplot of the original data
plt.scatter(x, y, label="Original Data", color="blue")

# Plot the regression lines for segment 1 and segment 2
segment_1_x, segment_1_predicted_y, model_1 = segment1
segment_2_x, segment_2_predicted_y, model_2 = segment2

# Calculate the confidence intervals for both segments
lower_1, upper_1 = Functions.confidence_interval(segment_1_x, y[:len(segment_1_x)], model_1)
lower_2, upper_2 = Functions.confidence_interval(segment_2_x, y[len(segment_1_x):], model_2)

set1 = (lower_1[-1], upper_1[-1])  # First set of points (vertical positions)
set2 = (lower_2[0], upper_2[0])  # Second set of points (vertical positions)

overlap_range = Functions.find_overlap_float(set1, set2)

if overlap_range:
    print("Confidence Intervals Overlap")
else:
    print("Confidence Intervals do not overlap")

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

