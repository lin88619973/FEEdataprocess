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

    # Get the channel data
    channel_data = data['channel_data']

    # Extract the digits from the file name as the label
    label = int(re.search(r'\d+', file).group())
    label = label * 50 / 1000

    # Split the channel data into columns
    channel_data_columns = np.hsplit(channel_data, channel_data.shape[1])

    # Clean the channel data columns by removing data points that deviate more than 0.1 times the mean value
    cleaned_channel_data_columns = []
    for column in channel_data_columns:
        column_mean = np.mean(column)
        column_std = np.std(column)
        threshold = 0.05 * column_mean
        cleaned_column = column[np.abs(column - column_mean) <= threshold]
        cleaned_channel_data_columns.append(cleaned_column)

    # Convert the cleaned channel data columns into the format (data, i)
    merged_data = []
    for i, column in enumerate(cleaned_channel_data_columns):
        merged_data.extend([(data, i) for data in column])

    # Merge all data into a new array
    merged_data = np.array(merged_data)

    # Extract labels and data points
    channels = merged_data[:, 1]
    data_points = merged_data[:, 0]

    # Calculate the mean and standard deviation for each channel
    mean_values = []
    std_values = []
    for channel in np.unique(channels):
        channel_data = data_points[channels == channel]
        mean_values.append(np.mean(channel_data))
        std_values.append(np.std(channel_data))

    # Plot a line graph
    plt.plot(np.unique(channels), mean_values, marker='o', markersize=3)
    plt.xlabel('Channel')
    plt.ylabel('ADC Value')
    plt.title('ADC Value of Each Channel')

    # Add a reference horizontal line
    mean_mean = np.mean(mean_values)
    plt.axhline(mean_mean, color='r', linestyle='-')

    # Save the line graph
    #save_path = os.path.join(parent_directory, 'pic', str(label) + 'channeltest.png')
    #plt.savefig(save_path)
#save_path = os.path.join(parent_directory, 'pic','channeltest.png')
plt.show()
