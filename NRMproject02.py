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
evap_rate = 0.2 # m3/s on the whole area 
A_cs_1 = 0.2463 # m^2

# Stakeholders data
# Power
power_demand = 10 # static MW
m_power_demand = power_demand + power_demand*0.1*np.cos(2*np.pi*np.arange(365)/365)**3
m_power_demand = np.array([m_power_demand for i in range(20)]).flatten() # Seasonal MW
Q_out_target = np.percentile(flow, 45)
# Environment
perc_25 = np.percentile(flow, 25) # static low pulse
perc_75 = np.percentile(flow, 75) # static high pulse
perc_95 = np.percentile(flow, 95) # static maximum target outflow
m_perc_25 = moving_perc(flow, sw=45, perc=25) # seasonal low pulse 
m_perc_75 = moving_perc(flow, sw=45, perc=75) # seasonal low pulse

# Alternatives: reservoir models
A0 = 1 # undammed river, no model required
A1 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 8,
		h_max_dam = 120, 
		h_max_flood = 110, 
		total_release_pipes_cross_section = A_cs_1*12, 
		initial_level = 0
		)
A21 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*5, 
		initial_level = 0
		)
A22 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*8, 
		initial_level = 0
		)
A23 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*12, 
		initial_level = 0
		)
A5 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 80, 
		h_max_flood = 5+68.47+2, 
		total_release_pipes_cross_section = A_cs_1*11, 
		initial_level = 0
		)

tested_reservoir = A1




# ######
res = A5
dh_target = 68.47
u_target_max = perc_75 / (A_cs_1*11*np.sqrt(2*9.81*dh_target))
#policy = operating_policy(h_max_flood=110, h_ds = 5, p=np.array([80, 0.1, 115]))
#A1.policy = policy

level = []
release = []
power = []
N = 4
block=True
for j in range(N):
	for t in range(-1, len(flow)-1):
		# at time t
		if block:
			u_valve=0
			if res.level >res.h_max_flood - 10 : block=False
		else:
			u_valve = res.get_policy_u()
			#u_valve = u_target_max
			#u_max =  perc_95 / (A_cs_1*11*np.sqrt(2*9.81*(res.level-5)))
			#u_max = u_max/(1.5*res.level - 1.5*res.h_max_flood) + u_valve
			#u_valve = u_max if (res.level >= res.h_max_flood) else u_valve
		res.update_valve(u_valve)
		# at time t+1
		res.update_level(flow[t+1], rain[t+1])
		level.append(res.level)
		release.append(res.release)
		power.append(res.get_power())

level = np.array(level)
release = np.array(release)
power = np.array(power)


fig, ax = plt.subplots(3)
ax[0].plot(level, label="level")
ax[0].plot([0, len(flow)*N], [res.h_max_dam, res.h_max_dam], "r--", alpha=0.8, label="Dam height")
ax[0].plot([0, len(flow)*N], [res.h_max_flood-2, res.h_max_flood-2], "g--", alpha=0.8, label="Target height")
ax[0].plot([0, len(flow)*N], [res.h_max_flood, res.h_max_flood], "--", c="orange", alpha=0.8, label="Flood line height")
ax[0].set_xlabel("Water level [m]")
ax[0].legend()
ax[0].grid()
ax[1].plot(release, "b", label="release")
ax[1].plot([0, len(flow)*N], [perc_25, perc_25], "--", c="orange", alpha=0.8, label="25th percentile")
ax[1].plot([0, len(flow)*N], [perc_75, perc_75], "r--", alpha=0.8, label="75th percentile")
ax[1].set_xlabel("Water release [$m^3/s$]")
ax[1].legend()
ax[1].grid()
ax[2].plot(power, "y", label="Power MW")
ax[2].plot(np.array([m_power_demand[:-1] for j in range(N)]).flatten(), "--",c="orange", alpha=0.8, label="Minimum power requirement")
ax[2].set_xlabel("Net output power [MW]")
ax[2].legend()
ax[2].grid()
plt.show()

print("Power index: ", Ipow_reliability(power[-7300:], m_power_demand) )


