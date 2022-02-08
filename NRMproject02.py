# python3.8
import numpy as np
import matplotlib
import matplotlib.pylab as plt

from NRMproject_utils0 import *
from NRMproject_plots import *
from NRMproject_utils3_reservoir import *


# Physical data
DATA_NAME = "C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\Natural_Resources_Management\\NRM_project_leck\\13Chatelot.csv"
DATA_NAME = "https://raw.githubusercontent.com/Matteoleccardi/2020NaturalresourcesManagement/main/13Chatelot.csv"
data  = np.loadtxt(DATA_NAME, delimiter=",", skiprows=1)
year  = data[:,0]
month = data[:,1]
day_m = data[:,2] # day number from month start (1-30)
day_y = np.array([range(1,365+1,1) for i in range(20)]).reshape((365*20, 1)) # day numbered from year start (1-365)
rain  = data[:,3] # mm/d
flow  = data[:,4] # m3/s daily average
temp  = data[:,5] # °C
area = 981.405000 # km^2

# Stakeholders data
water_demand = 100 # m3/d
h_flood = 2 # m

# Alternatives: reservoir models




# 

res = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = 2,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = 1.77*2, 
		initial_level = 0
		)

level = []
release = []
power = []
N = 4
for j in range(N):
	for i in range(len(flow)):
		if res.level >= res.h_max_flood-5:
			u_release = np.min( [(res.level - res.h_max_flood+5) / 10, 1])
		else:
			u_release = 0
		res.update_release(u_release)
		res.update_level(flow[i], rain[i])
		level.append(res.level)
		release.append(res.release)
		power.append(res.get_power())

plt.subplots()
plt.plot([0, len(flow)*N], [res.h_max_dam, res.h_max_dam], "r--")
plt.plot(level, label="level")
plt.plot(release, alpha=0.4, label="release")
plt.plot(power, alpha=0.5, label="power MW")
plt.legend()
plt.show()


