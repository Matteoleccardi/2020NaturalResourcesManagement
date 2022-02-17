# python3.8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import copy

from NRMproject_plots import getRGBA, getFC
### RESERVOIR MODEL ###

class reservoir():
	def __init__(self, surface_area, evaporation_rate, h_min, h_max_dam, h_max_flood, total_release_pipes_cross_section, initial_level, policy_params=None):
		# Static physical features
		self.surface_area = surface_area # m2
		self.evaporation_rate = evaporation_rate # m3 / s
		# Static design features
		self.h_min = h_min # m
		self.h_max_flood = h_max_flood #m
		self.h_max_dam = h_max_dam #m
		self.release_cs_max = total_release_pipes_cross_section # m2
		self.dead_capacity   = self.surface_area * self.h_min
		self.active_capacity = self.surface_area * (self.h_max_flood - self.h_min)
		self.flood_capacity  = self.surface_area * (self.h_max_dam - self.h_max_flood)
		self.capacity = self.surface_area * self.h_max_dam
		# Initial conditions
		self.level0 = initial_level # m
		self.release_cs0 = self.release_cs_max # m2
		self.release0 = 0 # m3/s
		self.overtop0 = 0
		self.v_out0 = 0 # m/s
		# State of the reservoir system at time t
		self.level = self.level0
		self.release_cs = self.release_cs0
		self.release = self.release0
		self.overtop = self.overtop0
		self.v_out = self.v_out0
		self.update_valve()
		# Utilities
		self.days_to_seconds = 24*60*60
		self.performance = 0
		# Operating policy related to the reservoir
		self.policy = operating_policy(self.h_max_flood, self.h_min, p=policy_params)

	def reset_initial_conditions(self):
		self.level = self.level0
		self.release_cs = self.release_cs0
		self.release = self.release0
		self.overtop = self.overtop0
		self.v_out = self.v_out0
		self.update_valve()

	def net_inflow(self, inflow, rain):
		rain = 0.001 * rain * self.surface_area / (self.days_to_seconds) # from mm/d to m3/s
		return inflow + rain - self.evaporation_rate

	def update_valve(self, u_release=1):
		u_release = np.min([np.max([u_release, 0]), 1])
		if self.level <= self.h_min:
			self.release = 0
		else:
			# Spillways contribution
			self.v_out = np.sqrt( 2*9.81*(self.level-self.h_min) )
			self.release_cs = u_release * self.release_cs_max
			self.release = self.release_cs * self.v_out # m3/s
			# Overtopping contribution
			self.overtop = np.max([self.level - self.h_max_dam, 0])
			if self.overtop > 0:
				self.release += 100*(self.overtop) + 10*(self.overtop)**2

	def set_release(self, target_release):
		''' Function to set the target release,
			changes the value of the valves accordingly
		'''
		return 0.5

	def update_level(self, inflow, rain):
		delta_level = (self.net_inflow(inflow, rain) - self.release) / self.surface_area
		delta_level = delta_level * self.days_to_seconds
		self.level = np.max([self.level+delta_level, 0])

	def get_power(self):
		W = 0.5 * 977 * self.release_cs * (self.v_out**3)
		W = 0.9 * W # efficiency about 90%
		if self.overtop > 0: W = 0
		return W/1e6 # MegaWatt = 10^6 J/s daily average

	def get_policy_u(self):
		return self.policy.get_u(self.level)

	def simulate(self, flow, rain, plot=False):
		level = []
		release = []
		power = []
		for t in range(len(flow)):
			# at time t
			u_valve = self.get_policy_u()	
			self.update_valve(u_valve)
			# step forward: t -> t+1
			self.update_level(flow[t], rain[t])
			level.append(self.level)
			release.append(self.release)
			power.append(self.get_power())
		measures = {
			"level":   np.array(level),
			"release": np.array(release),
			"power":   np.array(power)
		}
		if plot:
			self.plot_simulation(measures, flow)
		return measures

	def plot_simulation(self, measures, inflow=None):
		color_array = getRGBA()
		fig, axs = plt.subplots(3)
		# Water Level
		axs[0].plot(measures["level"], label="Level", color=color_array[16])
		axs[0].plot([0, len(measures["level"])], [self.h_max_dam, self.h_max_dam], "--", c=color_array[1], alpha=0.95, label="Dam height")
		axs[0].plot([0, len(measures["level"])], [self.h_max_flood, self.h_max_flood], "--", c=color_array[19], alpha=0.8, linewidth=0.8, label="Flood threshold")
		axs[0].plot([0, len(measures["level"])], [self.policy.h_max, self.policy.h_max], "--", c=color_array[6], alpha=0.8, linewidth=0.8, label="Policy max.")
		axs[0].plot([0, len(measures["level"])], [self.policy.h_min, self.policy.h_min], "--", c=color_array[6], alpha=0.8, linewidth=0.8, label="Policy min.")
		axs[0].set_ylabel("Water level (m)")
		axs[0].grid(linestyle='--', linewidth=0.5)
		axs[0].set_facecolor(getFC())
		axs[0].legend()
		# Release and natual streamflow
		if inflow is not None:
			axs[1].plot(inflow, label="Inflow", color=color_array[8], alpha=0.8, linewidth=0.8)
		axs[1].plot(measures["release"], label="Release", color=color_array[18])
		if inflow is not None:
			p = np.percentile(inflow, 0.25)
			axs[1].plot([0, len(measures["level"])], [p, p], "--", c="orange", alpha=0.5, linewidth=0.6, label="Inflow 25th, 50th, 75th perc.")
			p = np.percentile(inflow, 0.50)
			axs[1].plot([0, len(measures["level"])], [p, p], "--", c="orange", alpha=0.5, linewidth=0.6)
			p = np.percentile(inflow, 0.75)
			axs[1].plot([0, len(measures["level"])], [p, p], "--", c="orange", alpha=0.5, linewidth=0.6)
			p = np.mean(inflow)
			axs[1].plot([0, len(measures["level"])], [p, p], "--", c="red", alpha=0.5, linewidth=0.6, label="Mean")
		axs[1].set_ylabel("Flow ($m^3/s$)")
		axs[1].grid(linestyle='--', linewidth=0.5)
		axs[1].set_facecolor(getFC())
		axs[1].legend()
		# Generated power
		axs[2].plot(measures["power"], label="Power", color=color_array[11])
		axs[2].set_xlabel("Simulation time (Days)")
		axs[2].set_ylabel("Generated power (MW)")
		axs[2].grid(linestyle='--', linewidth=0.5)
		axs[2].set_facecolor(getFC())
		axs[2].legend()
		# Figure title
		l = int( len(measures["level"])/365 )
		fig.suptitle(f"Dam simulation over {l} years", fontsize=13)
		# Finalise plot
		plt.tight_layout()
		plt.show()

### OPERATING POLICIES DEFINITION ###

class operating_policy():
	def __init__(self, h_max_flood, h_ds = 5, p=None):
		self.h_min = h_ds + 2
		self.h_max = h_max_flood + 1
		if p is None:
			self.uniform_shuffle_params()
		elif len(p) == 3:
			self.update_params(p)
		else:
			print("Wrong number of parameters in operating policy.")
			quit()

	def get_u(self, level):
		u = 0.0
		if (level > self.h_min) and (level <= self.p[0]):
			u = ( (self.p[1]-0)/(self.p[0]-self.h_min) )*(level-self.h_min)+0
		if (level > self.p[0]) and (level <= self.p[2]):
			u = self.p[1]
		if (level > self.p[2]) and (level <= self.h_max):
			u = ( (1-self.p[1])/(self.h_max-self.p[2]) )*(level-self.p[2])+self.p[1]
		if level > self.h_max:
			u = 1.0
		return u

	def update_params(self, p):
		if len(p) == 3:
			p[2] = np.max(
				[np.min([p[2], self.h_max]), self.h_min+0.1]
			)
			p[1] = np.max( [np.min([p[1], 1]), 0] )
			p[0] = np.max(
				[np.min([p[0], p[2]]), self.h_min+0.001]
			)
			self.p = np.array(p)
		else:
			print("Wrong number of parameters while updating operating policy.")
			quit()

	def uniform_shuffle_params(self):
		x2 = np.random.random_sample()*(self.h_max-self.h_min-0.1) + self.h_min+0.1
		y  = np.random.random_sample()
		x1 = np.random.random_sample()*(x2-self.h_min) + self.h_min
		self.p = np.array([x1, y, x2])

	def normal_shuffle_params(self, variance=0.3):
		x2 = np.random.normal(self.p[2], variance*(self.h_max-self.h_min))
		while (x2 >= self.h_max) or (x2 <= self.h_min+0.1):
			x2 = np.random.normal(self.p[2], variance*(self.h_max-self.h_min))
		y = np.random.normal(self.p[1], variance)
		while (y > 1) or (y < 0):
			y = np.random.normal(self.p[1], variance)
		x1 = np.random.normal(np.min([self.p[0],x2]), variance*(x2-self.h_min))
		while (x1 >= x2):
			x1 = np.random.normal(np.min([self.p[0],x2]), variance*(x2-self.h_min))
		x1 = np.max([x1, self.h_min+0.01])
		self.p = np.array([x1, y, x2])

	def plot_policy(self, color_=None):
		if color_ is None:
			colorArray = getRGBA()
			color_ = colorArray[19]
		fig, ax = plt.subplots()
		ax.set_xlim([self.h_min-5, self.h_max+5])
		ax.set_ylim([-0.1, 1.1])
		ax.grid()
		x = np.array([self.h_min-10, self.h_min, self.p[0], self.p[2], self.h_max, self.h_max+10])
		y = np.array([0, 0, self.p[1], self.p[1], 1, 1])
		ax.plot(x, y, ".-", color=color_)
		ax.plot([self.h_min, self.h_min],[-0.1, 1.1], "--", color=colorArray[5])
		ax.plot([self.h_max, self.h_max],[-0.1, 1.1], "--", color=colorArray[5])
		ax.set_xlabel("Reservoir level [m]")
		ax.set_ylabel("Valve release [%]")
		ax.set_title("Operating policy of the release valve")
		ax.set_facecolor(getFC())
		plt.show()

	def plot_many_policies_setup(self, NumPolicies):
		fig, ax = plt.subplots()
		colorArray = getRGBA(NumPolicies)
		return fig, ax, colorArray

	def plot_many_policies(self, ax, plotIndex, color_array=None, pareto_indices=None, tot_policies=None):
		if color_array is None:
			color_ = "blue"
		else:
			color_ = color_array[plotIndex]
			if pareto_indices is not None:
				if np.sum(plotIndex == pareto_indices) != 0:
					i = int(tot_policies * (2/20))
					color_ = color_array[i]
					alpha_=0.8
				else:
					i = int(tot_policies * (6/20))
					color_ = color_array[i]
					alpha_=0.4
				cmap = matplotlib.cm.get_cmap('Spectral')
				legend_elements = [matplotlib.lines.Line2D([0], [0], color=cmap(0.3), label='Others' ),
		                           matplotlib.lines.Line2D([0], [0],  color=cmap(0.1), label='Pareto')
		        ]
				ax.legend(handles=legend_elements, loc='upper left')
		x = np.array([self.h_min-10, self.h_min, self.p[0], self.p[2], self.h_max, self.h_max+10])
		y = np.array([0, 0, self.p[1], self.p[1], 1, 1])
		ax.plot(x, y, ".-", color=color_, alpha=alpha_)
		ax.plot([self.h_min, self.h_min],[-0.1, 1.1], "--", color="orange", linewidth=0.5, alpha=alpha_)
		ax.plot([self.h_max, self.h_max],[-0.1, 1.1], "--", color="orange", linewidth=0.5, alpha=alpha_)
		ax.set_xlim([self.h_min-5, self.h_max+5])
		ax.set_ylim([-0.1, 1.1])
		ax.grid()
		ax.set_xlabel("Reservoir level [m]")
		ax.set_ylabel("Valve release [%]")
		ax.set_title(f"Operating policy of the release valve")
		ax.set_facecolor(getFC())
			
	



### GENETIC ALGORITHM ###

class population():
	def __init__(self,
		base_model,
		indices_list,
		indices_params_list,
		indices_inputs_list,
		N_individuals=10,
		mutation_probability=0.1,
		mutation_variance=0.5,
		indices_for_selection_list=[0]):
		# define population and basic population model
		self.base_individual = copy.deepcopy(base_model)
		self.N_individuals = N_individuals
		self.mutation_probability = mutation_probability
		self.mutation_variance = mutation_variance
		self.indices_list = indices_list
		self.indices_params_list = indices_params_list
		self.indices_inputs_list = indices_inputs_list
		self.indices_for_selection_list = indices_for_selection_list
		self.gen_performance = []
		self.N_generations=0
		self.all_pareto_idxs = []
		# Initialise population
		self.population = []
		for i in range(self.N_individuals):
			indiv = copy.deepcopy(self.base_individual)
			indiv.policy.uniform_shuffle_params()
			self.population.append( indiv )

	def get_matingprob(self, len_=None):
		if len_ == None:
			len_ = self.N_individuals
		if len_ > 400:
			return 0.0025
		elif len_ < 5:
			return 0.999
		else:
			param = np.array([-1.24853343e+03,  4.54035818e+03, -6.66885475e+03, 5.06717621e+03, -2.12759604e+03,  4.92845505e+02, -6.31921546e+01,  6.37384508e+00] )
			poly = np.poly1d(param)
			roots = (poly - np.log(len_)).roots
			root = np.real(roots[-1])
			root = np.min([np.max([root, 0.0025]),0.99])
			return root

	def test(self, flow, rain):
		performance = [] #list: performance[individual ID][Performance indices]
		for n in range(self.N_individuals):
			# Simulate system
			level = []
			release = []
			power = []
			for t in range(len(flow)):
				# at time t
				u_valve = self.population[n].get_policy_u()	
				self.population[n].update_valve(u_valve)
				# step forward: t -> t+1
				self.population[n].update_level(flow[t], rain[t])
				level.append(self.population[n].level)
				release.append(self.population[n].release)
				power.append(self.population[n].get_power())
			measures = {
				"level":   np.array(level),
				"release": np.array(release),
				"power":   np.array(power)
			}
			# Append performances related to individual n
			perf_list_temp = []
			for i in range(len(self.indices_list)):
				inp = measures[self.indices_inputs_list[i]][:]
				par = self.indices_params_list[i]
				perform = self.indices_list[i](inp, par)
				perf_list_temp.append( perform )
			performance.append(perf_list_temp)
			self.population[n].performance = np.array(perf_list_temp).copy()
		self.gen_performance.append( np.array(performance) ) #[n_generation][idx_individual,idx_Index]

	def apply_selection(self, selection_type="top half", indices_for_selection=[0]):
		''' after this method, the population will be ranked in order of best performance '''
		if selection_type == "all":
			if len(indices_for_selection) == 1:
				''' single objective '''
				rank_idx = np.argsort(self.gen_performance[-1][:,indices_for_selection[0]])
				rearranged_pop = []
				for idx in rank_idx:
					rearranged_pop.append(self.population[idx])
				self.population = rearranged_pop
				self.curr_pareto_idxs = rank_idx[0]
			elif len(indices_for_selection) >= 2:
				''' multi objective '''
				cost = np.array(self.gen_performance[-1][:,indices_for_selection])
				pareto_idx = self.is_pareto_efficient(cost, return_mask = False)
				rearranged_pop = []
				for idx in pareto_idx:
					rearranged_pop.append(self.population[idx])
				for i in range(len(self.population)):
					if np.sum(pareto_idx == i) == 0:
						rearranged_pop.append(self.population[i])
					if len(rearranged_pop) == len(self.population):
						break
				self.population = rearranged_pop
				self.curr_pareto_idxs = pareto_idx
			else:
				quit()
		elif selection_type == "top half":
			if len(indices_for_selection) == 1:
				''' single objective '''
				rank_idx = np.argsort(self.gen_performance[-1][:,indices_for_selection[0]])
				top_half_idx = rank_idx[:int(self.N_individuals/2)]
				rearranged_pop = []
				for idx in top_half_idx:
					rearranged_pop.append(self.population[idx])
				self.population = rearranged_pop
				self.curr_pareto_idxs = rank_idx[0]
			elif len(indices_for_selection) >= 2:
				''' multi objective '''
				cost = np.array(self.gen_performance[-1][:,indices_for_selection])
				pareto_idx = self.is_pareto_efficient(cost, return_mask = False)
				rearranged_pop = []
				for idx in pareto_idx:
					rearranged_pop.append(self.population[idx])
				for i in range(len(self.population)):
					if np.sum(pareto_idx == i) == 0:
						rearranged_pop.append(self.population[i])
					if len(rearranged_pop) == int(len(self.population)/2):
						break
				self.population = rearranged_pop
				self.curr_pareto_idxs = pareto_idx
			else:
				quit()		
		else:
			quit()
		self.all_pareto_idxs.append(self.curr_pareto_idxs)

	def is_pareto_efficient(self, costs, return_mask = True):
		"""
		Find the pareto-efficient points
		:param costs: An (n_points, n_costs) array
		:param return_mask: True to return a mask
		:return: An array of indices of pareto-efficient points.
		If return_mask is True, this will be an (n_points, ) boolean array
		Otherwise it will be a (n_efficient_points, ) integer array of indices.
		"""
		is_efficient = np.arange(costs.shape[0])
		n_points = costs.shape[0]
		next_point_index = 0  # Next index in the is_efficient array to search for
		while next_point_index<len(costs):
			nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
			nondominated_point_mask[next_point_index] = True
			is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
			costs = costs[nondominated_point_mask]
			next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
		if return_mask:
			is_efficient_mask = np.zeros(n_points, dtype = bool)
			is_efficient_mask[is_efficient] = True
			return is_efficient_mask
		else:
			return is_efficient

	def apply_mating(self, n_partners=2):
		n_partners = np.min( [np.max([n_partners, 2]) , len(self.population)-1] )
		n_pareto = len(self.curr_pareto_idxs)
		p_geom = self.get_matingprob(len_=len(self.population))
		dim = len(self.population[0].policy.p)
		new_population = []
		''' make the dominant ones survive '''
		n_always_mating = int( np.max([3, np.round(self.N_individuals*0.1)]) )
		self.n_survivors = np.min([self.N_individuals-n_always_mating, len(self.curr_pareto_idxs)])
		for i in range(self.n_survivors):
			# save new dominant individual
			new_individual = copy.deepcopy(self.base_individual)
			new_individual.policy.update_params(self.population[i].policy.p)
			new_individual.performance = self.population[i].performance.copy()
			new_population.append( new_individual )
		''' use the remaining population slots to fill up with new offspring produced by the best individuals '''
		for i in range(self.n_survivors,self.N_individuals):
			# select n partners
			prob = np.empty((0,))
			for j in range(n_partners):
				extraction = 1e09
				while extraction >= len(self.population):
					extraction = np.random.geometric(p=p_geom, size=1)
					if extraction < len(self.curr_pareto_idxs):
						extraction = np.floor( np.random.random_sample()*len(self.curr_pareto_idxs) )
				prob = np.append(prob, int(extraction))
			parents = []
			for p in prob:
				parents.append( self.population[int(p)] )
			# weight parents contributions
			prob_exchange = np.random.random_sample((dim,n_partners))
			for ii in range(dim):
				prob_exchange[ii,:] /= np.sum(prob_exchange[ii,:]) # normalise each parent conribution to parameter p[0]
			# mate and have child
			p_child = np.zeros(dim)
			for i in range(n_partners):
				p_child += prob_exchange[:,i]*parents[i].policy.p
			# save new individual
			new_individual = copy.deepcopy(self.base_individual)
			new_individual.policy.update_params(p_child)
			new_population.append( new_individual )
		self.population = new_population

	def get_nnn_closest_max_distance_dominance_set(self, n, Num=2):
		if len(self.curr_pareto_idxs) > 5:
			dist = []
			for i in range(len(self.curr_pareto_idxs)):
				if i == n:
					d=0
				else:
					d = np.linalg.norm(self.population[n].performance-self.population[i].performance)
					dist.append(d)
			dist = np.sort(np.array(dist))
			return dist[Num]
		else:
			return 0

	def get_mutation_indices(self):
		self.mutation_indices = np.arange(self.n_survivors,self.N_individuals)
		if len(self.curr_pareto_idxs) < self.N_individuals*0.80:
			''' the dominant set is too little, wait until it becomes crowded '''
			return 0
		# for each individual compute the mean distance of the 5 nearest neighbours is the dominant set
		m = []
		for n in range(len(self.curr_pareto_idxs)):
			m.append( self.get_nnn_closest_max_distance_dominance_set(n, Num=3) )
		m = np.array(m)
		# Uniformity detection
		if (np.max(m)-np.min(m))/2 < 1.2*np.median(m):
			''' the distribution of distances is uniform enough '''
			return 0
		# get the threshold
		m_thr = np.percentile(m, 35)
		# get indices
		m[m>m_thr] = 1e20
		num_under_thr = np.sum(m<1e19)
		if not (num_under_thr == 0):
			positions = np.argsort(m)[:num_under_thr]
			self.mutation_indices = np.append(self.mutation_indices, positions)

	def apply_mutation(self, mutation_type="gaussian"):
		self.get_mutation_indices()
		for i in self.mutation_indices:
			if np.random.random_sample() < self.mutation_probability:
				if mutation_type == "punctual gaussian":
					draw = int(np.round( np.random.random_sample()*3-0.5 ))
					p = self.population[i].policy.p
					var = 0.5 if draw == 1 else 15
					p[draw] = np.random.normal(p[draw], self.mutation_variance*var)
					self.population[i].policy.update_params(p)
				elif mutation_type == "punctual random":
					draw = int(np.round( np.random.random_sample()*3-0.5 ))
					p = self.population[i].policy.p
					if draw == 1:
						var = 1; add=0
					else:
						var=(self.population[i].policy.h_max-self.population[i].policy.h_min)
						add=self.population[i].policy.h_min
					p[draw] = np.random.random_sample()*self.mutation_variance*var + add
					self.population[i].policy.update_params(p)
				elif mutation_type == "gaussian":
					self.population[i].policy.normal_shuffle_params(variance=self.mutation_variance)
				elif mutation_type == "random":
					self.population[i].policy.uniform_shuffle_params()
				else:
					quit()

	def fully_evolve(self,
		flow,
		rain,
		N_generations=10,
		selection_type="top half",
		indices_for_selection=None,
		n_partners=2,
		mutation_type="gaussian",
		saveFigures=False):
		self.N_generations = N_generations
		if indices_for_selection is None:
			indices_for_selection=self.indices_for_selection_list
		# Setup interactive plot
		plt.ion()
		fig_pol, ax_pol, col_pol = self.population[0].policy.plot_many_policies_setup(self.N_individuals)
		fig_par, ax_par = plt.subplots(1)
		# cycle variables
		perf0=0
		# cycle through the generations
		for g in range(self.N_generations-1):
			print("Generation ", g+1)
			self.test(flow, rain)
			self.apply_selection(selection_type=selection_type, indices_for_selection=indices_for_selection)
			self.apply_mating(n_partners=n_partners)
			self.apply_mutation(mutation_type=mutation_type)
			# Dynamic parameters
			
			# Plot all policies of this generation
			ax_pol.clear()
			for i in range(self.N_individuals):
				self.population[i].policy.plot_many_policies(ax_pol, plotIndex=i, color_array=col_pol, pareto_indices=self.curr_pareto_idxs, tot_policies=self.N_individuals)
			ax_pol.set_title(f"Operating policy of the release valve (Gen {g+1})")
			# Plot performance index and pareto front of first 2 objectives
			self.plot_pareto_synchronous(ax_par, g)
			# Draw plots
			plt.pause(5)
			plt.draw()
			# Save figures
			if saveFigures:
				fig_par.savefig(f"./pareto_{g:04}.jpeg", bbox_inches="tight", dpi=150)
				fig_pol.savefig(f"./policies_{g:04}.jpeg", bbox_inches="tight", dpi=150)
		# Last generation needs just to be tested
		print("Last generation ", self.N_generations)
		self.test(flow, rain)
		self.apply_selection(selection_type=selection_type, indices_for_selection=indices_for_selection)
		self.gen_performance = np.array(self.gen_performance)
		# Plot all policies of this generation
		ax_pol.clear()
		for i in range(self.N_individuals):
			self.population[i].policy.plot_many_policies(ax_pol, plotIndex=i, color_array=col_pol, pareto_indices=self.curr_pareto_idxs, tot_policies=self.N_individuals)
		ax_pol.set_title(f"Operating policy of the release valve (Gen {self.N_generations})")
		# Plot performance index and pareto front of first 2 objectives
		self.plot_pareto_synchronous(ax_par, self.N_generations)
		# Draw plots
		plt.pause(0.5)
		plt.draw()
		# Save figures
		if saveFigures:
			fig_par.savefig(f"./pareto_{self.N_generations:04}.jpeg", bbox_inches="tight", dpi=150)
			fig_pol.savefig(f"./policies_{self.N_generations:04}.jpeg", bbox_inches="tight", dpi=150)
		# Last plot and messages
		plt.ioff()
		print("Evolutionary training completed.")

	def plot_pareto_synchronous(self, ax, generation_index):
		ax.clear()
		colArray = getRGBA()
		if generation_index-1 >= 0:
			ax.scatter(self.gen_performance[-2][:,0], self.gen_performance[-2][:,1], alpha=0.9, color=colArray[19], label="Gen. t-1")
		ax.scatter(self.gen_performance[-1][:,0], self.gen_performance[-1][:,1], alpha=0.9, color=colArray[6], label="Gen. t")
		ax.scatter(self.gen_performance[-1][self.curr_pareto_idxs,0], self.gen_performance[-1][self.curr_pareto_idxs,1], alpha=0.9, color=colArray[0], label="Pareto (Gen. t)")
		ax.grid()
		ax.set_xlabel("Objective 1")
		ax.set_ylabel("Objective 2")
		ax.set_title(f"Pareto frontier plot of generation {generation_index+1} of {self.N_generations}")
		ax.legend()
		ax.set_facecolor(getFC())

	def plot_pareto_all_generations(self):
		self.scatter_fig, self.scatter_ax = plt.subplots(1)
		plt.subplots_adjust(bottom=0.2)
		xmin = np.amin( self.gen_performance[:,:,0] )
		xmax = np.amax( self.gen_performance[:,:,0] )
		ymin = np.amin( self.gen_performance[:,:,1] )
		ymax = np.amax( self.gen_performance[:,:,1] )
		allowed_steps = np.arange(1,self.N_generations+1)
		ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02]) # left, bottom, width, height
		# create the sliders
		slider_gen = Slider(
			ax_slider, "Gen:", 1, self.N_generations,
			valinit=self.N_generations, valstep=allowed_steps,
			color=getRGBA()[19]
		)
		slider_gen.on_changed(self.update_plot_pareto_all_generations)
		# make first scatter
		col_t_1   = 1.0
		col_t     = 0.25
		col_t_par = 0.0
		self.scatter_ax.grid()
		self.scatter_ax.set_xlabel("Objective 1")
		self.scatter_ax.set_ylabel("Objective 2")
		self.scatter_ax.set_facecolor(getFC())
		self.scatter_ax.set_xlim([xmin, xmax])
		self.scatter_ax.set_ylim([ymin, ymax])
		# Create custom legend
		cmap = matplotlib.cm.get_cmap('Spectral')
		legend_elements = [matplotlib.lines.Line2D([0], [0], marker='o', color=cmap(col_t_1), label='Previous gen.',
                            markerfacecolor=cmap(col_t_1), markersize=5),
                           matplotlib.lines.Line2D([0], [0], marker='o', color=cmap(col_t), label='Current gen.',
                            markerfacecolor=cmap(col_t), markersize=5),
                           matplotlib.lines.Line2D([0], [0], marker='o', color=cmap(col_t_par), label='Pareto (current)',
                            markerfacecolor=cmap(col_t_par), markersize=5)
        ]
		self.scatter_ax.legend(handles=legend_elements, loc='upper right')
		# Put data and colors
		gen = self.N_generations
		par_idxs = self.all_pareto_idxs[gen-1][:]
		data = np.vstack([self.gen_performance[gen-2][:,:2],self.gen_performance[gen-1][:,:2],self.gen_performance[gen-1][par_idxs,:2]])
		colors = np.concatenate(
			[np.repeat(col_t_1,self.N_individuals) ,np.repeat(col_t,self.N_individuals) , np.repeat(col_t_par, len(self.all_pareto_idxs[gen-1][:]))]
		).flatten()
		self.scatter_ax.set_title(f"Pareto frontier plot of generation {gen-1}")
		self.scatter = self.scatter_ax.scatter(data[:,0], data[:,1], c=colors, alpha=0.9, cmap=cmap)
		self.scatter_ax.set_title(f"Pareto frontier plot of generation {gen}")
		self.scatter_fig.canvas.draw_idle()
		plt.show()
		
	def update_plot_pareto_all_generations(self, gen):
		col_t_1   = 1.0
		col_t     = 0.25
		col_t_par = 0.0
		label_t_1 = "Gen. " + str(int(gen-1))
		label_t = "Gen. " + str(int(gen))
		label_t_par = "Pareto"
		if gen-1 <= 0:
			par_idxs = self.all_pareto_idxs[0][:]
			data = np.vstack([self.gen_performance[0][:,:2],self.gen_performance[0][par_idxs,:2]])
			colors = np.concatenate(
				[np.repeat(col_t,self.N_individuals) , np.repeat(col_t_par, len(self.all_pareto_idxs[0][:]))]
			).flatten()
			self.scatter_ax.set_title(f"Pareto frontier plot of generation {1}")
		else:
			par_idxs = self.all_pareto_idxs[gen-1][:]
			data = np.vstack([self.gen_performance[gen-2][:,:2],self.gen_performance[gen-1][:,:2],self.gen_performance[gen-1][par_idxs,:2]])
			colors = np.concatenate(
				[np.repeat(col_t_1,self.N_individuals) ,np.repeat(col_t,self.N_individuals) , np.repeat(col_t_par, len(self.all_pareto_idxs[gen-1][:]))]
			).flatten()
			self.scatter_ax.set_title(f"Pareto frontier plot of generation {gen}")
		self.scatter.set_offsets(data)
		self.scatter.set_array(colors)
		self.scatter_fig.canvas.draw_idle()







### INDICATORS ###
''' the lower, the better the performance'''

# Water supply for irrigation: reliability, vulnerability, resilience
''' w is water demand expressed as m3/s daily mean '''
def Iirr_reliability(x, w):
	out = np.sum( x >= w ) / len(x)
	return 0-out
def Iirr_vulnerability(x, w, risk_aversion=1):
	y = w - x
	y[y<0] = 0
	num = np.sum( y ) ** risk_aversion
	den = np.sum( x < w )
	res = 0 if den == 0 else num / den
	return res
def Iirr_resilience(x, w):
	den = x[:-1] < w
	num = np.sum( den and (x[1:] >= w) )
	den = np.sum(x[:-1] < w)
	res = 1 if den == 0 else num / den
	return 0-res

# Water supply for hydroelectric power
def Ipow_Avg(power, target_power=0):
	''' power is not additive like i.e. water demand '''
	m = np.mean(power)
	return target_power-m
def I_pow_RMSE_from_setpoint(power, power_setpoint):
	return np.sqrt( np.mean( (power-power_setpoint)**2 ) )
def Ipow_reliability(power, pow_demand):
	out = np.sum( power >= pow_demand ) / len(power)
	return 0-out
def Ipow_vulnerability(power, pow_demand, risk_aversion=1):
	num = np.sum( np.max(pow_demand - power, 0) ) ** risk_aversion
	den = np.sum( power < pow_demand )
	res = 0 if den == 0 else num / den
	return res
def Ipow_resilience(power, pow_demand):
	den = power[:-1] < pow_demand
	num = np.sum( den and (power[1:] >= pow_demand) )
	den = np.sum(power[:-1] < pow_demand)
	res = 1 if den == 0 else num / den
	return 0-res

# Flooding indicators
''' h_max is the flood threshold '''
def Iflood_yearly_avg(h, hmax):
	n_years = len(h)/365
	out = np.sum( h >= hmax ) / n_years
	return out

def Iflood_max_area(h, hmax):
	steepness = 0.087 # steepness of 5 degrees
	x = (h-hmax)/steepness
	x_max = np.max([x, 0])
	return x_max

def Iflood_kayakClub(h, h_active_storage):
	h_club = h_active_storage + 3 # m
	n_years = len(h)/365
	out = np.sum( h >= h_club ) / n_years
	return out

# Environmental indicators
def Ienv_low_pulses(release, LP):
	''' LP is the 25 percentile of the natural system '''
	n_years = len(release)/365
	out = np.sum( release < LP ) / n_years
	return out
def Ienv_high_pulses(release, HP):
	''' HP is the 75 percentile of the natural system '''
	n_years = len(release)/365
	out = np.sum( release > HP ) / n_years
	return out
''' For the optimization algorithm, use these two as environmental indicators, not the two above '''
def Ienv_low_pulses_mean(release, LP):
	''' HP is the 75 percentile of the natural system '''
	if np.sum(release < LP) == 0:
		return 0-LP
	m = np.mean( release[release < LP] )
	return 0-m
def Ienv_high_pulses_mean(release, HP):
	''' HP is the 75 percentile of the natural system '''
	if np.sum(release > HP) == 0:
		return 0
	m = np.mean( release[release > HP] )
	return m















### OTHER UTILITIES ###

def moving_perc(flow, sw=20, perc=25):
	flow1 = flow[-sw:]
	flow1 = np.append(flow1, flow)
	flow1 = np.append(flow1, flow[:sw])
	fp = []
	for i in range(sw, len(flow)+sw):
		fp.append( np.percentile(flow1[i-sw:i+sw], perc) )
	return np.array(fp)


def powerDemand_to_heightDemand(A_cs_1 = 1.77, N=1, area=981.405000):
	A_cs = A_cs_1 * N
	power_demand_yearly_dynamic = 1000000*Power_demand + 5*np.cos(2*np.pi*np.arange(365)/365)**3
	pd = np.array([power_demand_yearly_dynamic for i in range(20)]).flatten()
	hd = 5 + ( (pd/(0.9*0.5*997*A_cs))**(2/3) - (101325/997) )/(2*9.81)
	print(np.max(hd), 1000*1000*area*np.max(hd))
	plt.plot(hd)
	plt.show()
	
def storage_capacity_design(flow):
	''' this is not afunction, it is just a place to store unused code '''
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