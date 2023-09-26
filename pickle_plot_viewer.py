import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog
import pickle

# Open a file dialog to choose the directory
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
root.destroy()

if file_path is None or file_path == "":
    print('No directory selected')
    exit()

# Load the figure from the saved file
with open(file_path, 'rb') as file:
    fig = pickle.load(file)

# Show the figure
plt.show()
