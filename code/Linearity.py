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

# Initialize a list to store all data points
all_data_points = []

# Iterate over all .mat files
for file in mat_files:
    # Read the .mat file
    file_path = os.path.join(folder_path, file)
    data = loadmat(file_path)
    
    # Extract the digits from the file name as the label
    label = int(re.search(r'\d+', file).group())
    label = label * 50 / 1000
    
    # Iterate over all variables in the .mat file
    for key, value in data.items():
        # Check if the value is a numerical array
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            # Pair the label with the data points to create a 2D array
            data_points = np.column_stack((np.full(value.size, label), value.flatten()))
            # Calculate the mean value of the data points
            mean_value = np.mean(data_points[:, 1])

            # Calculate the deviation of the data points from the mean value
            deviation = np.abs(data_points[:, 1] - mean_value)
            # Set the threshold as 10% of the mean value
            threshold = 0.1 * mean_value
            # Filter the data points based on whether the deviation is less than or equal to the threshold
            filtered_points = data_points[deviation <= threshold]

            # Add the filtered data points to the list
            all_data_points.append(filtered_points)

# Merge all the 2D arrays into one large 2D array
combined_data = np.vstack(all_data_points)
print(combined_data)

# Extract the labels and data points
labels = combined_data[:, 0]
data_points = combined_data[:, 1]

# Calculate the mean and standard deviation for each label
mean_values = []
std_values = []
for label in np.unique(labels):
    label_data = data_points[labels == label]
    mean_values.append(np.mean(label_data))
    std_values.append(np.std(label_data))

# Plot the error bar chart
plt.errorbar(np.unique(labels), mean_values, yerr=std_values, ms=4, ecolor='r', fmt='o', elinewidth=5, capsize=4)
plt.xlabel('Energy(pC)')
plt.ylabel('ADC value')
plt.title('')
plt.show()
