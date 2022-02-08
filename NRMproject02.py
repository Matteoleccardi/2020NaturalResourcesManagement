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
m_perc_25 = moving_perc(flow, sw=45, perc=25) # seasonal low pulse 
m_perc_75 = moving_perc(flow, sw=45, perc=75) # seasonal low pulse

##
secPerMonth = 60*60*24*30
target_r_monthly = Q_out_target * secPerMonth
flow_month = []
for i in range(12*19+9):
	flow_month.append( np.mean(flow[30*i:30*i+30])*secPerMonth )
plt.plot(flow_month)
plt.plot([0, len(flow_month)], [target_r_monthly,target_r_monthly])
plt.show()
s = 0
s_dyn = []
for i in range(len(flow_month)):
	s = s + (-flow_month[i] + target_r_monthly)
	if s < 0: s=0
	s_dyn.append(s)
s_dyn = np.array(s_dyn)
plt.figure()
plt.plot(s_dyn)
plt.show()
print("K optimal: ", np.max(s_dyn)*1e-09, "km^3")
print("h optimal: ", np.max(s_dyn)/(area*1e06))
quit()



# Alternatives: reservoir models
A0 = 1 # undammed river, no model required
A1 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 8,
		h_max_dam = 120, 
		h_max_flood = 110, 
		total_release_pipes_cross_section = A_cs_1*3, 
		initial_level = 0
		)
A21 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*1, 
		initial_level = 0
		)
A22 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*2, 
		initial_level = 0
		)
A23 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 74, 
		h_max_flood = 70, 
		total_release_pipes_cross_section = A_cs_1*3, 
		initial_level = 0
		)
A51 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 90, 
		h_max_flood = 85, 
		total_release_pipes_cross_section = A_cs_1*1, 
		initial_level = 0
		)
A52 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 60, 
		h_max_flood = 55, 
		total_release_pipes_cross_section = A_cs_1*2, 
		initial_level = 0
		)
A53 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 50,
		h_max_flood = 45, 
		total_release_pipes_cross_section = A_cs_1*3, 
		initial_level = 0
		)
A54 = reservoir(
		surface_area = area*1000*1000,
		evaporation_rate = evap_rate,
		h_min = 5,
		h_max_dam = 40, 
		h_max_flood = 35, 
		total_release_pipes_cross_section = A_cs_1*4, 
		initial_level = 0
		)

tested_reservoir = A1




# ######
res = A22
policy = operating_policy(h_max_flood=110, h_ds = 5, p=np.array([80, 0.1, 115]))
A1.policy = policy

level = []
release = []
power = []
N = 4
block=True
for j in range(N):
	for t in range(-1, len(flow)-1):
		# at time t
		u_release = res.get_policy_u()
		if block:
			u_release=0
			if res.level >res.h_max_flood - 5 : block=False
		# at time t+1
		res.update_release(u_release)
		res.update_level(flow[t+1], rain[t+1])
		level.append(res.level)
		release.append(res.release)
		power.append(res.get_power())

level = np.array(level)
release = np.array(release)
power = np.array(power)


plt.subplots()
plt.plot([0, len(flow)*N], [res.h_max_dam, res.h_max_dam], "r--")
plt.plot(level, label="level")
plt.plot(release, alpha=0.4, label="release")
plt.plot(power, alpha=0.5, label="power MW")
plt.legend()
plt.show()

print("Power index: ", Ipow_reliability(power[-7300:], m_power_demand) )


