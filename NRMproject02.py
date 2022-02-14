# python3.8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
from scipy.stats import planck

from NRMproject_utils0 import *
from NRMproject_plots import *
from NRMproject_utils3_reservoir import *

# Get data - online or offline
try:
	DATA_NAME = "C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\Natural_Resources_Management\\NRM_project_leck\\13Chatelot.csv"
	data  = np.loadtxt(DATA_NAME, delimiter=",", skiprows=1)
except Exception:
	try:
		DATA_NAME = "https://raw.githubusercontent.com/Matteoleccardi/2020NaturalresourcesManagement/main/13Chatelot.csv"
		data  = np.loadtxt(DATA_NAME, delimiter=",", skiprows=1)
	except Exception:
		print("ERROR: The source of the data cannot be found.")
		print("#####  Please check your internet connection\n#####  (data gets downloaded from server),")
		print("#####  or modify the source code and\n#####  insert the full path to data in the variable \"DATA_NAME\".")
		print("")
		quit()

# Physical data
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

# EMODPS - EVOLUTIONARY MULTY OBJECTIVE DIRECT POLICY SEARCH

# Model to test
tested_reservoir = A5
# Population data
N_individuals = 50
# Total generations
N_generations = 15
# inputs
N_iterations = 3
ext_flow = np.array([flow for i in range(N_iterations)]).flatten().copy()
ext_rain = np.array([rain for i in range(N_iterations)]).flatten().copy()
# objectives
indices_list=[Ipow_Avg, Ienv_high_pulses_mean]
indices_params_list=[30, perc_75]
indices_inputs_list=["power", "release"]
# optimization
indices_for_selection_list = [0, 1] # first element of "indices_list"
# selection strategy
selection_type="top half" #["all","top half"]
# mating strategy
n_partners=2
# mutation strategy
mutation_type="random"
mutation_prob = 0.01
mutation_variance = 0.5


pop = population(
	base_model=tested_reservoir,
	N_individuals=N_individuals,
	mutation_probability=mutation_prob,
	mutation_variance=mutation_variance,
	indices_list=indices_list,
	indices_params_list=indices_params_list,
	indices_inputs_list=indices_inputs_list,
	indices_for_selection_list=indices_for_selection_list
	)

pop.fully_evolve(
	N_generations=N_generations,
	flow=ext_flow,
	rain=ext_rain,
	selection_type=selection_type,
	n_partners=n_partners,
	mutation_type=mutation_type
	)


plt.close('all')
pop.plot_pareto_all_generations()







quit()
'''
# Initial population
N_individuals = 30
population = []
for i in range(N_individuals):
	tested_reservoir.policy.uniform_shuffle_params()
	population.append( copy.deepcopy(tested_reservoir) )
# first run
performance = []
power_setpoint = 40
for n in range(N_individuals): # Simulate individuals
	# Simulate system
	level = []
	release = []
	power = []
	for t in range(len(ext_flow)):
		# at time t
		u_valve = population[n].get_policy_u()	
		population[n].update_valve(u_valve)
		# at time t+1
		f = ext_flow[t]
		r = ext_rain[t]
		population[n].update_level(f, r)
		level.append(population[n].level)
		release.append(population[n].release)
		power.append(population[n].get_power())
	level = np.array(level)
	release = np.array(release)
	power = np.array(power)
	performance.append(objective(power, power_setpoint))

# selection
rank_idx = np.argsort(performance)
best = rank_idx[:int(N_individuals/2)]
# mating
new_population = []
for i in range(3):
	new_individual = tested_reservoir
	new_individual.policy.update_params(population[best[i]].policy.p)
	new_population.append( copy.deepcopy(new_individual) )
for i in range(3, N_individuals):
	[prob1, prob2] = mating_prob.rvs(size=2)
	prob1 = prob1 if prob1<len(best) else prob1=len(best)
	prob2 = prob2 if prob2<len(best) else prob2=len(best)
	p1 = population[best[prob1]].policy.p
	p2 = population[best[prob2]].policy.p
	prob_exchange = np.random.random_sample((3,))
	p_child = prob_exchange*p1 + (1-prob_exchange)*p2
	new_individual = tested_reservoir
	new_individual.policy.update_params(p_child)
	new_population.append( copy.deepcopy(new_individual) )
# mutation
for i in range(len(new_population)):
	if np.random.random_sample() < mutation_prob
		new_population[i].policy.normal_shuffle_params(self, variance=mutation_valiance)

# update population and repeat

plt.plot(performance, '.-')
plt.show()





quit()
'''
# ######
res = A5
policy = operating_policy(h_max_flood=res.h_max_flood, h_ds = 5, p=np.array([68, 0.5, 5+68+2]))
res.policy = policy

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


