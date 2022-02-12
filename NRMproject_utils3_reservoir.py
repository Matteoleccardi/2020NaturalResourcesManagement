# python3.8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
		# State of the reservoir system at time t
		self.level = initial_level # m
		self.release_cs = self.release_cs_max # m2
		self.release = 0 # m3/s
		self.overtop = 0
		self.v_out = 0 # m/s
		self.update_valve()
		# Utilities
		self.days_to_seconds = 24*60*60
		# Operating policy related to the reservoir
		self.policy = operating_policy(self.h_max_flood, self.h_min, p=policy_params)

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
		return 17.0

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

	def plot_many_policies(self, ax, plotIndex, color_array=None):
		if color_array is None:
			color_ = "blue"
		else:
			color_ = color_array[plotIndex]
		x = np.array([self.h_min-10, self.h_min, self.p[0], self.p[2], self.h_max, self.h_max+10])
		y = np.array([0, 0, self.p[1], self.p[1], 1, 1])
		ax.plot(x, y, ".-", color=color_, alpha=0.5)
		ax.plot([self.h_min, self.h_min],[-0.1, 1.1], "--", color="orange", linewidth=0.8, alpha=0.8)
		ax.plot([self.h_max, self.h_max],[-0.1, 1.1], "--", color="orange", linewidth=0.8, alpha=0.8)
		ax.set_xlim([self.h_min-5, self.h_max+5])
		ax.set_ylim([-0.1, 1.1])
		ax.grid()
		ax.set_xlabel("Reservoir level [m]")
		ax.set_ylabel("Valve release [%]")
		ax.set_title("Operating policy of the release valve")
		ax.set_facecolor(getFC())
			
	



### GENETIC ALGORITHM ###

class population():
	def __init__(self,
		base_model,
		mating_probability_distribution,
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
		self.mating_probability_distribution = mating_probability_distribution
		self.mutation_probability = mutation_probability
		self.mutation_variance = mutation_variance
		self.indices_list = indices_list
		self.indices_params_list = indices_params_list
		self.indices_inputs_list = indices_inputs_list
		self.indices_for_selection_list = indices_for_selection_list
		self.gen_performance = []
		self.N_generations=0
		# Initialise population
		self.population = []
		for i in range(self.N_individuals):
			indiv = copy.deepcopy(self.base_individual)
			indiv.policy.uniform_shuffle_params()
			self.population.append( indiv )

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
				inp = measures[self.indices_inputs_list[i]][-365:]
				par = self.indices_params_list[i]
				perform = self.indices_list[i](inp, par)
				perf_list_temp.append( perform )
			performance.append(perf_list_temp)
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

	def apply_mating(self, n_partners=2, n_survivors=3):
		if n_survivors >= len(self.population):
			n_survivors = len(self.population)-1
		n_partners = np.min( [np.max([n_partners, 2]) , len(self.population)-1] )
		new_population = []
		for i in range(n_survivors):
			survivor = copy.deepcopy(self.base_individual)
			survivor.policy.update_params(self.population[i].policy.p)
			new_population.append( survivor )
		for i in range(n_survivors, self.N_individuals):
			# select parents
			prob = self.mating_probability_distribution.rvs(size=n_partners)
			for j in range(len(prob)):
				prob[j] = prob[j] if (prob[j] < len(self.population)) else len(self.population)-1
			parents = []
			for p in prob: parents.append( self.population[p] )
			# weight parents contributions
			prob_exchange = np.random.random_sample((3,n_partners))
			prob_exchange[0,:] /= np.sum(prob_exchange[0,:]) # normalise each parent conribution to parameter p[0]
			prob_exchange[1,:] /= np.sum(prob_exchange[1,:]) # normalise each parent conribution to parameter p[1]
			prob_exchange[2,:] /= np.sum(prob_exchange[2,:]) # normalise each parent conribution to parameter p[2]
			# mate and have child
			p_child = np.zeros(3)
			for i in range(n_partners):
				p_child += prob_exchange[:,i]*parents[i].policy.p
			# save new individual
			new_individual = copy.deepcopy(self.base_individual)
			new_individual.policy.update_params(p_child)
			new_population.append( new_individual )
		self.population = new_population

	def apply_mutation(self, mutation_type="gaussian"):
		for i in range(self.N_individuals):
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
		n_survivors=3,
		mutation_type="gaussian"):
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
			self.apply_mating(n_partners=n_partners, n_survivors=n_survivors)
			self.apply_mutation(mutation_type=mutation_type)
			# Dynamic parameters
			
			# Plot all policies of this generation
			ax_pol.clear()
			for i in range(self.N_individuals):
				self.population[i].policy.plot_many_policies(ax_pol, plotIndex=i, color_array=col_pol)
			# Plot performance index and pareto front of first 2 objectives
			self.plot_pareto_synchronous(ax_par, g)
			# Draw plots
			plt.pause(0.5)
			plt.draw()
		# Last generation needs just to be tested
		print("Generation ", self.N_generations)
		self.test(flow, rain)
		self.apply_selection(selection_type=selection_type, indices_for_selection=indices_for_selection)
		self.gen_performance = np.array(self.gen_performance)
		# Last plots
		plt.ioff()

	def plot_pareto_synchronous(self, ax, generation_index):
		ax.clear()
		colArray = getRGBA()
		if generation_index-1 >= 0:
			ax.scatter(self.gen_performance[generation_index-1][:,0], self.gen_performance[generation_index-1][:,1], alpha=0.6, color=colArray[19], label="Gen. t-1")
		ax.scatter(self.gen_performance[generation_index][:,0], self.gen_performance[generation_index][:,1], alpha=0.6, color=colArray[6], label="Gen. t")
		ax.scatter(self.gen_performance[generation_index][self.curr_pareto_idxs,0], self.gen_performance[generation_index][self.curr_pareto_idxs,1], alpha=0.2, color=colArray[0], label="Pareto (Gen. t)")
		ax.grid()
		ax.set_xlabel("Objective 1")
		ax.set_ylabel("Objective 2")
		ax.set_title(f"Pareto frontier plot of generation {generation_index+1}")
		ax.legend()
		ax.set_facecolor(getFC())

	def plot_pareto_all_generations(self):
		plt.ion()
		fig, ax = plt.subplots(1)
		xmin = np.amin( self.gen_performance[:,:,0] )
		xmax = np.amax( self.gen_performance[:,:,0] )
		ymin = np.amin( self.gen_performance[:,:,1] )
		ymax = np.amax( self.gen_performance[:,:,1] )
		length = self.gen_performance.shape[0]
		#
		generation_index = 0
		while generation_index < length:
			ax.clear()
			colArray = getRGBA()
			if generation_index-1 >= 0:
				ax.scatter(self.gen_performance[generation_index-1][:,0], self.gen_performance[generation_index-1][:,1], alpha=0.6, color=colArray[19], label="Gen. t-1")
			ax.scatter(self.gen_performance[generation_index][:,0], self.gen_performance[generation_index][:,1], alpha=0.6, color=colArray[6], label="Gen. t")
			ax.scatter(self.gen_performance[generation_index][self.curr_pareto_idxs,0], self.gen_performance[generation_index][self.curr_pareto_idxs,1], alpha=0.2, color=colArray[0], label="Pareto (Gen. t)")
			ax.grid()
			ax.set_xlabel("Objective 1")
			ax.set_ylabel("Objective 2")
			ax.set_title(f"Pareto frontier plot of generation {generation_index+1}")
			ax.legend()
			ax.set_facecolor(getFC())
			ax.set_xlim([xmin, xmax])
			ax.set_ylim([ymin, ymax])
			plt.draw()
			plt.pause(0.5)
			##
			generation_index += 1
			if generation_index == length:
				generation_index = 0
		plt.ioff()

		return 0








### INDICATORS ###
''' the lower, the better '''

# Water supply for irrigation: reliability, vulnerability, resilience
''' w is water demand expressed as m3/s daily mean '''
def Iirr_reliability(x, w):
	out = np.sum( x >= w ) / len(x)
	return out
def Iirr_vulnerability(x, w, power=1):
	num = np.sum( np.max(w - x, 0) ) ** power
	den = np.sum( x < w )
	if den == 0:
		res = 0
	else:
		res = num / den
	return res
def Iirr_resilience(x, w):
	den = x[:-1] < w
	num = np.sum( den and (x[1:] >= w) )
	den = np.sum(x[:-1] < w)
	if den == 0:
		res = 0
	else:
		res = num / den
	return res

# Water supply for hydroelectric power
def Ipow_Avg(power, lambda_=30):
	''' power is not additive like i.e. water demand '''
	m = np.mean(power)
	return np.exp(-m/lambda_)

def I_pow_RMSE_from_setpoint(power, power_setpoint):
	return np.sqrt( np.mean( (power-power_setpoint)**2 ) )

def Ipow_reliability(power, pow_demand):
	out = np.sum( power >= pow_demand ) / len(power)
	return out

def Ipow_vulnerability(power, pow_demand, risk_aversion=1):
	num = np.sum( np.max(pow_demand - power, 0) ) ** risk_aversion
	den = np.sum( power < pow_demand )
	if den == 0:
		res = 0
	else:
		res = num / den
	return res

def Ipow_resilience(power, pow_demand):
	den = power[:-1] < pow_demand
	num = np.sum( den and (power[1:] >= pow_demand) )
	den = np.sum(power[:-1] < pow_demand)
	if den == 0:
		res = 0
	else:
		res = num / den
	return res

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
	out = np.sum( h >= h_active_storage ) / n_years
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

def Ienv_high_pulses_mean(release, HP, lambda_=40):
	''' HP is the 75 percentile of the natural system '''
	m = np.mean( release[release> HP] )+0.001
	return 1-np.exp(-m/lambda_)















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