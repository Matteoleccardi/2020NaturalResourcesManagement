# python3.8
import numpy as np
import matplotlib
import matplotlib.pylab as plt


### RESERVOIR MODELS ###

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




class operating_policy():
	def __init__(self, h_max_flood, h_ds = 5, p=None):
		self.h_min = h_ds + 2
		self.h_max = h_max_flood + 1
		if p is None:
			x2 = np.random.random_sample()*(self.h_max-self.h_min)+ self.h_min
			y  = np.random.random_sample()
			x1 = np.random.random_sample()*(self.h_max-self.h_min)+ self.h_min
			while x1 >= x2:
				x1 = np.random.random_sample()*(self.h_max-self.h_min)+ self.h_min
			self.p = np.array([x1, y, x2])
		elif len(p) == 3:
			if p[2] >= self.h_max:
				p[2] = self.h_max - np.random.random_sample()
			if p[2] <= self.h_min:
				p[2] = self.h_max - np.random.random_sample()*(self.h_max-self.h_min)/2
			p[1] = np.max( [np.min([p[1], 1]), 0] )
			if (p[0] <= self.h_min) or (p[0] >= self.h_max):
				p[0] = self.h_min + np.random.random_sample()
				while p[0] >= p[2]:
					p[0] = self.h_min + np.random.random_sample()
			self.p = np.array([p[0], p[1], p[2]]) 
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

	def update_params(self):
		self.p = self.p












### INDICATORS ###

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
def Ipow_Avg(power):
	''' power is not additive like i.e. water demand '''
	return np.mean(power)

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
	out = np.sum( x > HP ) / n_years
	return out















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