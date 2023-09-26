# VibroVision
Event-based camera vibration analysis algorithms.

NOTE: Due to metavision only having drivers for Windows, mac users may need to remove all metavision SDK code in order to use.

The following packages are required to use this software:
1. Charidotella
2. SciPy
3. NumPy
4. OpenCV (requires python 3.8.6 environment)
5. Metavision SDK
6. Pandas
7. matplotlib
8. pickle
9. tkinter

The mainAutomate.py is used to define the parameters for the algorithms, then by running the script you will be asked to select the folder with the recordings to be analysed. once selected, the code will run for all recordings and save the results as pickle plots. the pickle_plot_viewer.py can be used to view these.
