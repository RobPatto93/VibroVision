import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle

from import_data import import_data
from scipy import ndimage, signal, interpolate, optimize
from skimage import morphology, io 
from scipy import fftpack

file_path= os.getenv('file_path')
sample_size = int(os.getenv('sample_size'))
start_event = int(os.getenv('start_event'))  #set the start event for the histogram
data_ingest = import_data(file_path,sample_size,start_event)

#camera resolution (resolution in the array + 1)
x_res = (np.max(data_ingest['x']))+1 
y_res = (np.max(data_ingest['y']))+1  

# set the value range for the first column
sample_rate = int(os.getenv('sample_rate'))     #sample rate is in us (microseconds)
lower_limit = int(os.getenv('start_event'))     #what number should the index start. useful if there is an issue with the beginning of the data
upper_limit = lower_limit + sample_rate         # sets the limits for hte boolean indexing


# boolean indexing to get rows where the first column satisfies the condition
mask = (data_ingest['t'] >= lower_limit) & (data_ingest['t'] <= upper_limit)

# create a new numpy array with the rows that satisfy the condition
sample_array = data_ingest[mask][['t','x','y','p']]

x_offset = 0                                             #pixel offset to the left of the image
y_offset = 0                                             #pixel offset to the top of the image
cell_width = 1                                           #pixel width of the cell (set to 1)
cell_height = 1                                          #pixel height of the cell (set to 1)
x_indices = (sample_array['x'] - x_offset) // cell_width #creates array of the x indices
y_indices = (sample_array['y']- y_offset) // cell_height #creates array of the y indices
event_frame = np.zeros([x_res,y_res])
for t in sample_array['t']:
    event_frame[x_indices,y_indices] = 1
    
event_frame = event_frame.T                              #transposes the matrix so the graphs are the right way up

errosion_factor = 3
selem = morphology.square(errosion_factor)
opened_ts = morphology.binary_erosion(event_frame, selem)

dilation_factor = 10
selem = morphology.square(dilation_factor)
closed_ts = morphology.binary_dilation(opened_ts, selem)

closed_ts = closed_ts.T                                  #transposes the matrix so the graphs are the right way up
center_of_mass = ndimage.center_of_mass(closed_ts)       #find the center of mass of the time surface, note this type is tuple
center_of_mass_stack = np.array(center_of_mass)          #convert the tuple to an array


#increment the timestamp ready for the main loop 
timestamp = upper_limit + sample_rate  

#create an array of timestamps to use in the main for i in range loop
timestamps = np.arange(data_ingest[0]['t'], data_ingest[-1]['t'], sample_rate)


# Main loop that increments by sample_rate until sample_rate is >= data_ingest[-1,0]
for i in range(len(timestamps)):
    # boolean indexing to get rows where the first column satisfies the condition
    mask = (data_ingest['t'] >= timestamp) & (data_ingest['t'] <= timestamp+sample_rate)

    # create a new numpy array with the rows that satisfy the condition
    sample_array = data_ingest[mask][['t','x','y','p']]

    # create 2D event frame
    x_indices = (sample_array['x'])         #creates array of the x indices
    y_indices = (sample_array['y'])         #creates array of the y indices
    event_frame = np.zeros([x_res,y_res])   #creates a zero matrix of the data_ingest resolution
    event_frame[x_indices,y_indices] = 1    #input all the events into the time surface 

                     ###noise filtering ###

    # perform morphological operations to filter the time surface
    selem = morphology.square(errosion_factor)
    opened_ts = morphology.binary_erosion(event_frame, selem)
    selem = morphology.square(dilation_factor)
    closed_ts = morphology.binary_dilation(opened_ts, selem)

    if np.any(closed_ts):
        center_of_mass = ndimage.center_of_mass(closed_ts)
        centroid_array = np.array(center_of_mass)
        center_of_mass_stack = np.vstack((center_of_mass_stack,centroid_array)) #stack the centroid arrays

    # update timestamp
    timestamp += sample_rate

# Remove rows containing NaN values
freq_plot = center_of_mass_stack[~np.isnan(center_of_mass_stack).any(axis=1)]

sample_hz = (sample_rate) / 1000000  # Sampling rate in Hz

x = freq_plot[:, 0]  # X coordinates of the centroid stack
y = freq_plot[:, 1]  # Y coordinates of the centroid stack

# Perform FFT on the x and y coordinates
x_fft = fftpack.fft(x)
y_fft = fftpack.fft(y)

# Set a threshold value
threshold = 0.0  # Adjust the threshold level as needed

# Apply thresholding
x_fft[np.abs(x_fft) < threshold] = 0.0
y_fft[np.abs(y_fft) < threshold] = 0.0

# Perform inverse FFT to get the denoised signals
x_denoised = fftpack.ifft(x_fft)
y_denoised = fftpack.ifft(y_fft)

# Create an array of frequencies
freqs = fftpack.fftfreq(len(y), d=sample_hz)

# Plot the FFT results
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

# Plot 1: FFT of x
ax1.stem(freqs, np.abs(x_fft))
ax1.set_xlabel('Frequency in Hertz [Hz]')
ax1.set_ylabel('Magnitude')
ax1.set_title('FFT of x (Before Thresholding)')
ax1.set_xlim(1, 100)
ax1.set_ylim(threshold, np.max(np.abs(x_fft)))

# Plot 2: FFT of y
ax2.stem(freqs, np.abs(y_fft))
ax2.set_xlabel('Frequency in Hertz [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('FFT of y (Before Thresholding)')
ax2.set_xlim(1, 100)
ax2.set_ylim(threshold, np.max(np.abs(y_fft)))

# Plot 3: Denoised Signal of x
ax3.plot((timestamps[:len(x_denoised)]/1000000),x_denoised)
ax3.set_xlabel('Time')
ax3.set_ylabel('Amplitude')
ax3.set_title('Denoised Signal of x')

# Plot 4: Denoised Signal of y
ax4.plot((timestamps[:len(y_denoised)]/1000000),y_denoised)
ax4.set_xlabel('Time')
ax4.set_ylabel('Amplitude')
ax4.set_title('Denoised Signal of y')

# Compute the magnitudes of x_fft and y_fft
x_magnitudes = np.abs(x_fft)
y_magnitudes = np.abs(y_fft)

# Get the indices of the top 10 frequencies by magnitude for x_fft
x_top_indices = np.argsort(x_magnitudes)[::-1][:10]
x_top_frequencies = freqs[x_top_indices]
x_top_magnitudes = x_magnitudes[x_top_indices]

#Get the indices of the top 10 frequencies by magnitude for y_fft
y_top_indices = np.argsort(y_magnitudes)[::-1][:10]
y_top_frequencies = freqs[y_top_indices]
y_top_magnitudes = y_magnitudes[y_top_indices]

# Combine ax5 and ax6 on the same plot
ax5.stem(x_top_frequencies, x_top_magnitudes, 'C1', label='x_fft', linefmt='blue', markerfmt='b')
ax5.stem(x_top_frequencies, y_top_magnitudes, 'C2', label='y_fft', linefmt='orange', markerfmt='orange')
ax5.set_xlabel('Frequency in Hertz [Hz]')
ax5.set_ylabel('Magnitude')
ax5.set_title('Top 10 Magnitudes')
ax5.set_xlim(0.5, None)  # Set the x-axis lower limit to 0.5
ax5.set_ylim(threshold, np.max(x_top_magnitudes))  # Set the y-axis limits
ax5.legend()

# Hide the unused plot
ax6.axis('off')

folder_name = 'centroid_pickle_plots'
os.makedirs(folder_name, exist_ok=True)

pickle_path = os.path.join(folder_name, file_path + 'CentroidFFT.pkl')
print('saved as: ' + pickle_path)

# Save the pickle plot to the absolute file path
with open(pickle_path, 'wb') as file:
    pickle.dump(fig, file)









