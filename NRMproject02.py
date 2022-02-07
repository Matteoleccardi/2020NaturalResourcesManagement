# python3.8
import numpy as np
import matplotlib
import matplotlib.pylab as plt

from NRMproject_utils0 import *
from NRMproject_utils1_ANN import *
from NRMproject_plots import *
from NRMproject_utils2_trees import *

DATA_NAME = "C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\Natural_Resources_Management\\NRM_project_leck\\13Chatelot.csv"
DATA_NAME = "https://raw.githubusercontent.com/Matteoleccardi/2020NaturalresourcesManagement/main/13Chatelot.csv"

data  = np.loadtxt(DATA_NAME, delimiter=",", skiprows=1)
year  = data[:,0]
month = data[:,1]
day_m = data[:,2] # day number from month start (1-30)
day_y = np.array([range(1,365+1,1) for i in range(20)]).reshape((365*20, 1)) # day numbered from year start (1-365)
rain  = data[:,3] # mm/d
flow  = data[:,4] # m3/d
temp  = data[:,5] # °C


