import math
import numpy as np
import matplotlib.pyplot as plt 
import rebound
import os
import csv
import sys
import statistics

from rebound import hash as h
from ipywidgets import interact
from matplotlib.animation import FuncAnimation, FFMpegWriter

def split_specific_sites():
    
    #***CONSTANTS*** (to be offloaded into a .info file or something......hardwired for TRAPPIST at the moment)
    num_planets = 7
    object_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    parent = os.getcwd()
    folder_paths = []
    bin_paths = []
    bin_slices = 101
    for i in range(1,21):
        folder_paths.append(os.path.join(parent, "5000e_1000y_0vinc_" + str(i)))
        bin_paths.append("5000e_1000y_0vinc_" + str(i) + "/5000e_1000y_0vinc_" + str(i)+ ".bin")

    colors = ('palegreen', 'w', 'w','r', 'c', 'm', 'gold', 'darkgrey', 'b', 'g')
    groups = ('remaining', 'escaped', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')

    Rearth_to_AU = 4.2635e-5
    km_to_AU = 6.68459e-9

    d_rad = 0.788*Rearth_to_AU

    semimaj = (1.154e-2, 1.580e-2, 2.227e-2, 2.925e-2, 3.849e-2, 4.683e-2, 6.189e-2)
