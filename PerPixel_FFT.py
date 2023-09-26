import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os

from import_data import import_data  # Replace "my_module" with the actual module name
from scipy import ndimage, signal, interpolate, optimize
from skimage import morphology, io 
from scipy import fftpack

#checks if data ingest has already occured and stops multiple loads to memory  
file_path = os.getenv('file_path')
filename = os.path.basename(file_path)
sample_size = int(os.environ['sample_size'])
start_event = int(os.getenv('start_event'))  #set the start event for the histogram
data_ingest = import_data(file_path,sample_size,start_event)
print(np.size(data_ingest))

#camera resolution (resolution in the array + 1)
x_res = (np.max(data_ingest['x']))+1  
y_res = (np.max(data_ingest['y']))+1   

# set the value range for the first column
target_number_pixels = 100
sample_pixels_for_target = 100000
sample_rate = 5000                                    #sample rate is in us (microseconds), <20hz = 20000, 20hz> = 10000 - 5000, >100hz = 10000
lower_limit = 0                                       #what number should the index start. useful if there is an issue with the beginning of the data
upper_limit = lower_limit + sample_pixels_for_target # sets the limits for hte boolean indexing
print(data_ingest['t'] >= lower_limit)
print(data_ingest['t'] <= upper_limit)

# boolean indexing to get rows where the first column satisfies the condition
mask = (data_ingest['t'] >= lower_limit) & (data_ingest['t'] <= upper_limit)

# create a new numpy array with the rows that satisfy the condition
sample_array = data_ingest[mask][['t','x','y','p']]

x_indices = sample_array['x']  # Array of x indices
y_indices = sample_array['y']  # Array of y indices
event_frame = np.zeros([x_res, y_res])

event_frame[x_indices, y_indices] = 1
    
event_frame = event_frame.T #transposes the matrix so the graphs are the right way up

errosion_factor = 1
selem = morphology.square(errosion_factor)
target_pixels = morphology.binary_erosion(event_frame, selem)

true_count = np.count_nonzero(target_pixels)
print("Number of Active Pixels: ", true_count)

while true_count >target_number_pixels:
    errosion_factor = errosion_factor + 1
    selem = morphology.square(errosion_factor)
    target_pixels = morphology.binary_erosion(event_frame, selem)
    true_count = np.count_nonzero(target_pixels)

print("Number of fft Pixels: ", true_count)
print('Errosion factor used:', errosion_factor)

# Create coordinate grids
y_coords, x_coords = np.indices(target_pixels.shape)

# Get the matching indices where target_pixels is True
matching_indices = np.where(target_pixels)

# Create an empty dictionary
matching_dict = {}

# Use broadcasting to assign the 't' values to the matching indices
for ingest in data_ingest:
    x = ingest['x']
    y = ingest['y']
    t = ingest['t']
    if target_pixels[y, x]:
        key = (x, y)
        if key not in matching_dict:
            matching_dict[key] = []
        matching_dict[key].append(t)

print('t values extracted into dictionary')

# Dictionary to store binned values
binned_dict = {}

# Iterate over the keys in matching_dict
for key, values in matching_dict.items():
    # Calculate the bin edges based on the sample rate
    bin_edges = np.arange(0, max(values) + sample_rate, sample_rate)

    # Perform the histogram binning
    counts, _ = np.histogram(values, bins=bin_edges)

    # Store the counts in binned_dict
    binned_dict[key] = counts.tolist()

print('t values binned into bin size of:' + str(sample_rate) + 'us')

# Initialize a dictionary to store the highest 10 magnitudes and frequencies for each key
top_10_dict = {}

# Find the maximum FFT size among all the values in binned_dict
fft_size = max(len(values) for values in binned_dict.values())

# Iterate over each key-value pair in binned_dict
for key, values in binned_dict.items():
    # Compute the FFT of the values with the maximum FFT size
    pixel_fft = fftpack.fft(values, n=fft_size)

    # Set a threshold value
    threshold = 0.0  # Adjust the threshold level as needed

    # Apply thresholding
    pixel_fft[np.abs(pixel_fft) < threshold] = 0.0

    # Perform inverse FFT to get the denoised signals
    y_denoised = fftpack.ifft(pixel_fft)

    # Create an array of frequencies
    freqs = fftpack.fftfreq(fft_size, d=sample_rate/1e6)

    # Find the frequencies and magnitudes for frequencies greater than 0.1 Hz and positive
    indices = np.where((freqs > 0.1) & (pixel_fft > 0))
    frequencies = freqs[indices]
    magnitudes = np.abs(pixel_fft)[indices]

    # Store the frequencies and magnitudes in the dictionary
    top_10_dict[key] = {'frequencies': frequencies, 'magnitudes': magnitudes}

# Plot the FFT results for all keys on a single plot
fig, ax = plt.subplots()

# Iterate over each key in top_10_dict
for key, data in top_10_dict.items():
    frequencies = data['frequencies']
    magnitudes = data['magnitudes']

    # Plot the frequencies and magnitudes for the current key
    ax.stem(frequencies, magnitudes)

ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Magnitude')
ax.set_title('Top 10 Magnitudes and Frequencies for '+ str(true_count) + ' Active Pixels')
# ax.legend()

# Define the folder name and create it if it doesn't exist
folder_name = 'pickle_plots_pixel__fft'
os.makedirs(folder_name, exist_ok=True)

pickle_path = os.path.join(folder_name, file_path + 'PerPixelFFT.pkl')
print('saved as: ' + pickle_path)

# Save the pickle plot to the absolute file path
with open(pickle_path, 'wb') as file:
    pickle.dump(fig, file)


