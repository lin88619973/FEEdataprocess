import os
import re
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

# Get the current directory of the file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path
parent_directory = os.path.dirname(current_directory)

# Concatenate the parent directory and folder name
folder_path = os.path.join(parent_directory, '4xdata')


# List all .mat files in the folder
mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

# Initialize lists to store file name numbers and means
file_numbers = []
means = []

# Iterate over all .mat files
for file in mat_files:
    # Load the .mat file
    data = loadmat(os.path.join(folder_path, file))
    
    # Initialize variables for mean calculation
    total_sum = 0
    count = 0
    
    # Iterate over the data dictionary and calculate the mean of all numeric elements
    for key, value in data.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            total_sum += value.sum()
            count += value.size
    
    # Calculate the mean
    if count > 0:
        mean_value = total_sum / count
    else:
        mean_value = np.nan  # Use NaN to represent if there are no numeric elements
    
    # Extract the number from the file name
    file_number = int(re.search(r'\d+', file).group())
    
    # Store the results
    file_numbers.append(file_number)
    means.append(mean_value)

# Filter out NaN values
valid_indices = np.isfinite(means)
file_numbers = np.array(file_numbers)[valid_indices]
means = np.array(means)[valid_indices]

print(file_numbers)
print(means)

# Create a 2D array
data_array = np.array([file_numbers, means]).T

# Sort the data_array in ascending order based on file_numbers
sorted_data_array = data_array[data_array[:, 0].argsort()]

# Update file_numbers and means
file_numbers = sorted_data_array[:, 0]
means = sorted_data_array[:, 1]
# Update file_numbers
file_numbers = file_numbers * 50 / 1000
# Only take the first xx data points for fitting
file_numbers_subset = file_numbers[:36]
means_subset = means[:36]
# Plot the scatter plot
plt.scatter(file_numbers, means, label='All data points')

# Linear fit the first xx data points
coefficients = np.polyfit(file_numbers_subset, means_subset, 1)

# Calculate the y values of the fitted line
x_line = np.linspace(min(file_numbers_subset), max(file_numbers_subset), 100)
y_line = np.polyval(coefficients, x_line)

# Plot the fitted line
plt.plot(x_line, y_line, 'r-', label='Linear fit')

# Add legend and labels
plt.legend()
plt.ylabel('ADC code')
# Modify the x-axis unit to pC
plt.xlabel('charge (pC)')
# Calculate the linearity of the fitted line
residuals = means_subset - np.polyval(coefficients, file_numbers_subset)
linear_fit_degree = np.std(residuals)
# Calculate R-squared
total_sum_of_squares = np.sum((means_subset - np.mean(means_subset))**2)
residual_sum_of_squares = np.sum(residuals**2)
r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
# Label the linearity and R-squared on the plot
plt.text(0.1, 0.9, f'Linear Fit Degree: {linear_fit_degree:.2f}', transform=plt.gca().transAxes)
#plt.text(0.1, 0.8, f'R-squared: {r_squared:.2f}', transform=plt.gca().transAxes)
# Display the plot
plt.show()
