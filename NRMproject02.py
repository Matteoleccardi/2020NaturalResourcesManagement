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
N_individuals = 120
# Total generations
N_generations = 50
# inputs
N_iterations = 3
ext_flow = np.array([flow for i in range(N_iterations)]).flatten().copy()
ext_rain = np.array([rain for i in range(N_iterations)]).flatten().copy()
# objectives
indices_list=[Ipow_Avg, Ienv_high_pulses_mean]
indices_params_list=[40, perc_75]
indices_inputs_list=["power", "release"]
# optimization
indices_for_selection_list = [0, 1] # element of "indices_list" to consider in selection algorithm
# selection strategy
selection_type="all" # "all" is the best one. Others: ["all","top half"]
# mating strategy
n_partners=2
# mutation strategy
mutation_prob = 0.3
mutation_type="random"
mutation_variance = 0.5


''' Run a simulation of the dam and plot results '''
if 1:
	# Flow and rain input can be modified at will
	f = ext_flow.copy()
	r = ext_rain.copy()
	# This simulation will leave the simulated model exactly as it was defined
	dam_to_sim = copy.deepcopy(A5)
	dam_to_sim.policy.update_params([dam_to_sim.policy.h_min+1, 0.999, dam_to_sim.policy.h_max-1])
	dam_to_sim.simulate(f, r, plot=True)

''' Run the EMODPS algorithm '''
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
	mutation_type=mutation_type,
	saveFigures=True
	)

''' See results of the policy evolution '''
plt.close('all')
pop.plot_pareto_all_generations()
# Simulate one of the results of the optimization
pop.population[0].reset_initial_conditions()
pop.population[0].simulate(ext_flow, ext_rain, plot=True)
