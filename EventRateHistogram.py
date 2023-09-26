import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from import_data import import_data
from scipy import ndimage, signal, interpolate, optimize
from skimage import morphology, io 
from scipy import fftpack

# read in data from a file
#file_path= 'data_sets/trans_20hz_2s.es'
file_path = os.getenv('file_path')
sample_size = int(os.getenv('sample_size'))
start_event = int(os.getenv('start_event'))  #set the start event for the histogram
data_ingest = import_data(file_path,sample_size,start_event)

#camera resolution (resolution in the array + 1)
x_res = (np.max(data_ingest['x']))           #set the resolution to the max value of the x column
y_res = (np.max(data_ingest['y']))           #set the resolution to the max value of the y column
sample_size = int(os.getenv('sample_size'))  #set the sample size for the histogram
bin_size = int(os.getenv('bin_size'))        #set the bin size for the histogram
start_event = int(os.getenv('start_event'))  #set the start event for the histogram

# set the value range for the first column
start_event = 0                       #what number should the index start. useful if there is an issue with the beginning of the data
end_event = start_event + sample_size # sets the limits for the boolean indexing

# boolean indexing to get rows where the first column satisfies the condition
mask = (data_ingest['t'] >= start_event) & (data_ingest['t'] <= end_event)

# create a new numpy array with the rows that satisfy the condition
sample_array = data_ingest[mask][['t', 'x', 'y', 'p']]  

#to run ON or ON&OFF events
mask = (sample_array['p'] >= 0) # to include on and off events, set to >= 0, only on >= 1, only off < 1. 
sample_array = sample_array[mask]

# ensure timestamps are sorted
timestamps = np.sort(sample_array['t'])

# calculate the differences between consecutive timestamps
diffs = np.diff(timestamps)

# convert diffs to int64 and count the number of occurrences of each difference
counts = np.bincount(diffs.astype('int64'))

# calculate the event rate (number of events per time period)
event_rate = counts / 0.001  # divide by the length of the time period to get the rate

hist_data = np.histogram(timestamps, bins=bin_size)  # Compute histogram data
bins = hist_data[1]
heights = hist_data[0]

# Create the histogram plot
plt.figure(figsize=(10, 6))
plt.bar(bins[:-1]/1000000, heights, width=np.diff(bins)/1000000)

plt.title('Histogram of Event Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Events')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

folder_name = 'pickle_plots'
os.makedirs(folder_name, exist_ok=True)

pickle_path = os.path.join(folder_name, file_path + 'EventRateHistogram.pkl')
print('saved as: ' + pickle_path)

# Save the pickle plot to the absolute file path
with open(pickle_path, 'wb') as file:
    pickle.dump(fig, file)

