import os
import numpy as np
from tkinter import Tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import subprocess

def filepath_generator():
    #create a list of all the filepaths that the below functions will be run on

    file_paths_list = []
    for root, directories, files in os.walk(directory):
        for file in files:
            if file.endswith('.raw'):
                file_paths_list.append(os.path.join(root, file).replace("\\","/"))
    return file_paths_list

def event_rate_histogram():          
    subprocess.run(["python","EventRateHistogram.py"])

def centroid_fft():
    subprocess.run(["python","CentroidFFT.py"])

def per_pixel_fft():
    subprocess.run(["python","PerPixelFFT.py"])

def run():

    #choose what to run
    run_event_rate_histogram = False
    run_centroid_fft = True
    run_per_pixel_fft = True

    #set the variables
    sample_size = 10000000
    sample_rate = 5000  # 10 000 for < 20 hz, 5000 for 20 - 40hz
    start_event = 3000000  #us
    bin_size = 500

    os.environ['sample_size'] = str(sample_size)
    os.environ['sample_rate'] = str(sample_rate)
    os.environ['start_event'] = str(start_event)
    os.environ['bin_size'] = str(bin_size)

    for filepaths in file_paths_list:
        os.environ['file_path'] = str(filepaths)
        print(filepaths)
        
        if run_event_rate_histogram is True:
            print("beginning event rate histogram")
            event_rate_histogram()
        if run_centroid_fft is True:
            print("beginning centroid fft")
            centroid_fft()
        if run_per_pixel_fft is True:
            print("beginning per pixel fft")
            per_pixel_fft()


# have a filedialouge box to choose the directory
root = Tk()
root.withdraw()
directory = filedialog.askdirectory()    
# run the filepath list generator function  
if directory is None:
    print('No directory selected')
    exit()     
file_paths_list = filepath_generator()
# run the main function

run()


