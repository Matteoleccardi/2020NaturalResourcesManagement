# python3.8
import numpy as np
import matplotlib
import matplotlib.pylab as plt


### RESERVOIR MODELS ###

class reservoir():
	def __init__(self, surface_area, evaporation_rate, h_min, h_max_dam, h_max_flood, total_release_pipes_cross_section, initial_level):
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
		self.update_release()
		# Utilities
		self.days_to_seconds = 24*60*60

	def net_inflow(self, inflow, rain):
		rain = 0.001 * rain * self.surface_area / (self.days_to_seconds) # from mm/d to m3/s
		return inflow + rain - self.evaporation_rate

	def update_release(self, u_release=1):
		if self.level <= self.h_min:
			self.release = 0
		else:
			self.v_out = np.sqrt( 2*(9.81*(self.level-self.h_min) + 100000/997) )
			self.release_cs = u_release * self.release_cs_max
			# Spillways contribution
			self.release = self.release_cs * self.v_out # m3/s
			# Overtopping contribution
			self.overtop = np.max([self.level - self.h_max_dam, 0])
			if self.overtop > 0:
				self.release += 15*(self.overtop)**2

	def update_level(self, inflow, rain):
		delta_level = (self.net_inflow(inflow, rain) - self.release) / self.surface_area
		delta_level = delta_level * self.days_to_seconds
		self.level = np.max([self.level+delta_level, 0])

	def get_power(self):
		W = 0.5 * 977 * self.release_cs * (self.v_out**3)
		W = 0.9 * W # efficiency about 90%
		if self.overtop > 0: W = 0
		return W/1e6 # MegaWatt = 10^6 J/s daily average













### INDICATORS ###

# Water supply for irrigation: reliability, vulnerability, resilience
''' w is water demand expressed as m3/s daily mean '''
def Iirr_reliability(x, w):
	if size(w) != 1: quit()
	out = np.sum( x >= w ) / len(x)
	return out
def Iirr_vulnerability(x, w, power=1):
	if size(w) != 1: quit()
	num = np.sum( np.max(w - x, 0) ) ** power
	den = np.sum( x < w )
	if den == 0:
		res = 0
	else:
		res = num / den
	return res
def Iirr_resilience(x, w):
	if size(w) != 1: quit()
	den = x[:-1] < w
	num = np.sum( den and (x[1:] >= w) )
	den = np.sum(x[:-1] < w)
	if den == 0:
		res = 0
	else:
		res = num / den
	return res


# Water supply for hydroelectric power


# Flooding indicators
''' h_max is the flood threshold '''
def Iflood_yearly_avg(h, hmax):
	if size(hmax) != 1: quit()
	n_years = len(h)/365
	out = np.sum( h >= hmax ) / n_years
	return out
def Iflood_max_area(h, hmax):
	steepness = 0.087 # steepness of 5 degrees
	x = (h-hmax)/steepness
	x_max = np.max([x, 0])
	return x_max

# Environmental indicators
def Ienv_low_pulses(x, LP):
	''' LP is the 25 percentile of the natural system '''
	if size(LP) != 1: quit()
	n_years = len(x)/365
	out = np.sum( x < LP ) / n_years
	return out
def Ienv_high_pulses(x, HP):
	''' HP is the 75 percentile of the natural system '''
	if size(HP) != 1: quit()
	n_years = len(x)/365
	out = np.sum( x > HP ) / n_years
	return out


