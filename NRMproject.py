# python3.8
import numpy as np
import scipy
import scipy.stats as ss
import matplotlib
import matplotlib.pylab as plt
import torch

from NRMproject_utils0 import *
from NRMproject_utils1_ANN import *
from NRMproject_plots import *

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

### RAW DATA VISUALIZATION
if 0 : plot_rawData(flow, rain, temp, year)

### STATIONARY STATISTICS
if 0 : printStationaryStats(flow, rain, temp)

### PROGRAM COMPARTMENTS
PART_10 = 0
PART_11 = 1





if PART_10:
	############ PART 1.0: MODEL LINEAR IN PARAMETERS' SPACE ###################

	### DATASETS
	w = 6

	# Streamflow datasets
	flow_DS  = Dataset00(flow, year, w)
	detr_flow = (flow_DS.series-flow_DS.ma) / np.sqrt(flow_DS.mv)
	flow1_DS = Dataset00(detr_flow, year, w)
	if 0: plot_seriesDetrending(flow_DS, flow1_DS, day_y, obj="streamflow")

	# Rain datasets
	rain_DS  = Dataset00(rain, year, w)
	r = (rain_DS.series-rain_DS.ma) / np.sqrt(rain_DS.mv)
	rain1_DS = Dataset00(r, year, w)
	if 0: plot_seriesDetrending(rain_DS, rain1_DS, day_y, obj="rainfall")


	# Temperature datasets
	temp_DS  = Dataset00(temp, year, w)
	r = (temp_DS.series-temp_DS.ma) / np.sqrt(temp_DS.mv)
	temp1_DS = Dataset00(r, year, w)
	if 0: plot_seriesDetrending(temp_DS, temp1_DS, day_y, obj="temperatures")






	### BASELINE MODEL: Yearly Moving Average
	lossesVal = []
	lossesValHighFlow = []
	lossesValLowFlow = []
	for val in range(20):
		# Calibration
		''' Calibration is performed automatically while instancing the Dataset00() class:
			moving average and moving variance for 19 of the 20 years
		'''
		# Validation over real world data
		obs = flow_DS.valid[val]
		obs_est = flow_DS.train_ma[val][:365]
		# Save losses with respect to validation dataset
		lossesVal.append( RMSEloss(obs, obs_est) )
		lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[:365], flow_DS.mv[:365]) )
		lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[:365], flow_DS.mv[:365]) )
	# Save losses with respect to model
	lossesN = totalLoss(np.array(lossesVal))
	lossesNhigh = totalLoss(np.array(lossesValHighFlow))
	lossesNlow = totalLoss(np.array(lossesValLowFlow))
	if 0:
		print("\nBaseline model (yearly MA) results: ")
		print("Baseline loss J: ", lossesN)
		print("Baseline loss J for high flows: ", lossesNhigh)
		print("Baseline loss J for medium-low flows: ", lossesNlow)







	### AR MODELS

	# Linear AR(N)
	Nmax = 2
	lossesN = []
	lossesNhigh = []
	lossesNlow = []
	series_DS = flow_DS
	series1_DS = flow1_DS
	for N in range(1,Nmax+1):
		lossesVal = []
		lossesValHighFlow = []
		lossesValLowFlow = []
		for val in range(20):
			# Calibration
			obs = series1_DS.train[val][N:]
			x   = series1_DS.train[val][0:-1]
			A = getA(x, N)
			p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
			# Validation over real world data
			obs = series_DS.valid[val][N:]
			x   = series1_DS.valid[val][0:-1]
			X = getX(x, N)
			obs_est = normToRaw(AR(p, X), series_DS.ma[N:365], series_DS.mv[N:365])
			# Save losses with respect to validation dataset
			lossesVal.append( RMSEloss(obs, obs_est) )
			lossesValHighFlow.append( RMSElossHigh(obs, obs_est, series_DS.ma[N:365], series_DS.mv[N:365]) )
			lossesValLowFlow.append( RMSElossLow(obs, obs_est, series_DS.ma[N:365], series_DS.mv[N:365]) )
		# Save losses with respect to model order
		lossesVal = np.array(lossesVal)
		lossesValHighFlow = np.array(lossesValHighFlow)
		lossesValLowFlow = np.array(lossesValLowFlow)
		lossesN.append( totalLoss(lossesVal) )
		lossesNhigh.append( totalLoss(lossesValHighFlow) )
		lossesNlow.append( totalLoss(lossesValLowFlow) )

	if 0:
		plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear AR(N) results:")
		plot_linearARlosses(lossesN, lossesNhigh, lossesNlow, add_title=": linear AR(N) models")


	# Polinomial AR(N)
	Nmax = 2
	lossesN = []
	lossesNhigh = []
	lossesNlow = []
	for N in range(1,Nmax+1):
		lossesVal = []
		lossesValHighFlow = []
		lossesValLowFlow = []
		for val in range(20):
			# Calibration
			obs = flow1_DS.train[val][N:]
			x   = flow1_DS.train[val][0:-1]
			A = getApoli(x, N)
			p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
			# Validation over real world data
			obs = flow_DS.valid[val][N:]
			x   = flow1_DS.valid[val][0:-1]
			X = getXpoli(x, N)
			obs_est = normToRaw(AR(p, X), flow_DS.ma[N:365], flow_DS.mv[N:365])
			# Save losses with respect to validation dataset
			lossesVal.append( RMSEloss(obs, obs_est) )
			lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[N:365], flow_DS.mv[N:365]) )
			lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[N:365], flow_DS.mv[N:365]) )
		# Save losses with respect to model order
		lossesVal = np.array(lossesVal)
		lossesValHighFlow = np.array(lossesValHighFlow)
		lossesValLowFlow = np.array(lossesValLowFlow)
		lossesN.append( totalLoss(lossesVal) )
		lossesNhigh.append( totalLoss(lossesValHighFlow) )
		lossesNlow.append( totalLoss(lossesValLowFlow) )

	if 0:
		#plot_autocorrelogram(obs - obs_est, order=60)
		print("\nPolinomial AR(N) results:")
		plot_linearARlosses(lossesN, lossesNhigh, lossesNlow, add_title=": polinomial AR(N) models")






	### ARX MODELS


	# Linear proper ARX(N, P) models: AR(N) + X(P)
	''' ARX(N, 1):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p[]*u(t) + p[]
		ARX(N, P):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p(N-1)*x(t-N+1)
		           + p[...]*u(t) + p[]*u(t-1)+ ... + p[]*u(t-P+1)
		           + p[-1] 
		u(t) is the standardised input (rain or temperature)
		p is the parameter vector coming out of the lstsq(): N+P+1
	'''
	Nmax = 1
	Pmax = 1
	lossesN     = np.zeros((Nmax,Pmax))  # caution: (index) of model is (degree of model -1)
	lossesNhigh = np.zeros((Nmax,Pmax))
	lossesNlow  = np.zeros((Nmax,Pmax))
	exog_DS = rain1_DS
	for N in range(1, Nmax+1):
		for P in range(1, Pmax+1):
			idxStart = int(np.max([N, P]))
			lossesVal         = []
			lossesValHighFlow = []
			lossesValLowFlow  = []
			# Cross-validation cycle
			for val in range(20):
				# Calibration
				obs = flow1_DS.train[val][idxStart:]
				x   = flow1_DS.train[val][0:-1]
				u   = exog_DS.train[val][0:-1]
				A = getAarx(x, N, u, P)
				p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
				# Validation over real world data
				obs = flow_DS.valid[val][idxStart:]
				x   = flow1_DS.valid[val][0:-1]
				u   = exog_DS.valid[val][0:-1]
				X = getXarx(x, N, u, P)
				obs_est = normToRaw(AR(p, X), flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365])
				# Save losses with respect to validation dataset
				lossesVal.append( RMSEloss(obs, obs_est) )
				lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
				lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
			# Save losses with respect to model order
			lossesVal         = np.array(lossesVal)
			lossesValHighFlow = np.array(lossesValHighFlow)
			lossesValLowFlow  = np.array(lossesValLowFlow)
			lossesN[N-1,P-1]     = totalLoss(lossesVal)
			lossesNhigh[N-1,P-1] = totalLoss(lossesValHighFlow)
			lossesNlow[N-1,P-1]  = totalLoss(lossesValLowFlow)

	if 0:
		plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear ARX(N, P) (proper rainfall) results:")
		plot_linearARXlosses(lossesN, lossesNhigh, lossesNlow, Nref=None, Pref=None)



	# Linear improper ARX(N, P) models: AR(N) + X(P+1)
	''' ARX(N, 1):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p[]*u(t) + p[]*u(t+1) + p[]
		ARX(N, P):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p(N-1)*x(t-N+1)
		           + p[...]*u(t) + p[]*u(t-1)+ ... + p[]*u(t-P+1) + p[]*u(t+1) 
		           + p[-1] 
		u(t) is the standardised input (rain or temperature)
		p is the parameter vector coming out of the lstsq(): N+P+1
	'''
	Nmax = 1
	Pmax = 1
	lossesN     = np.zeros((Nmax,Pmax))  # caution: (index) of model is (degree of model -1)
	lossesNhigh = np.zeros((Nmax,Pmax))
	lossesNlow  = np.zeros((Nmax,Pmax))
	exog_DS = rain1_DS
	for N in range(1, Nmax+1):
		for P in range(1, Pmax+1):
			idxStart = int(np.max([N, P]))
			lossesVal         = []
			lossesValHighFlow = []
			lossesValLowFlow  = []
			# Cross-validation cycle
			for val in range(20):
				# Calibration
				obs   = flow1_DS.train[val][idxStart:]
				x     = flow1_DS.train[val][0:-1]
				u     = exog_DS.train[val][0:-1]
				u_imp = exog_DS.train[val][idxStart:]
				A = getAarxImp(x, N, u, P, u_imp)
				p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
				# Validation over real world data
				obs   = flow_DS.valid[val][idxStart:]
				x     = flow1_DS.valid[val][0:-1]
				u     = exog_DS.valid[val][0:-1]
				u_imp = exog_DS.valid[val][idxStart:]
				X = getXarxImp(x, N, u, P, u_imp)
				obs_est = normToRaw(AR(p, X), flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365])
				# Save losses with respect to validation dataset
				lossesVal.append( RMSEloss(obs, obs_est) )
				lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
				lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
			# Save losses with respect to model order
			lossesVal         = np.array(lossesVal)
			lossesValHighFlow = np.array(lossesValHighFlow)
			lossesValLowFlow  = np.array(lossesValLowFlow)
			lossesN[N-1,P-1]     = totalLoss(lossesVal)
			lossesNhigh[N-1,P-1] = totalLoss(lossesValHighFlow)
			lossesNlow[N-1,P-1]  = totalLoss(lossesValLowFlow)

	if 0:
		plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear ARX(N, P+1) (improper rainfall) results:")
		plot_linearARXlosses(lossesN, lossesNhigh, lossesNlow, Nref=None, Pref=None)



	# Linear ARX(N, P, T) models with proper X(P) and proper temperture T(T): AR(N) + X(P) + T(T)
	''' ARX(N, 1):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p[]*u(t) + p[]*T(t, t-1, t-2, ..., t-T+1) + p[N]
		ARX(N, P):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p(N-1)*x(t-N+1) +
		           + p[...]*u(t) + p[]*u(t-1)+ ... + p[]*u(t-P+1) +
		           + p[]*T(t, t-1, t-2, ..., t-T+1) +
		           + p[-1] 
		u(t) is the standardised input (rain or temperature)
		p is the parameter vector coming out of the lstsq(): N+P+1
	'''
	Nmax = 1
	Pmax = 1
	T    = 1 # T must remain fixed and > 0
	lossesN     = np.zeros((Nmax,Pmax))  # caution: (index) of model is (degree of model -1)
	lossesNhigh = np.zeros((Nmax,Pmax))
	lossesNlow  = np.zeros((Nmax,Pmax))

	for N in range(1, Nmax+1):
		for P in range(1, Pmax+1):
			idxStart = int(np.max([N, P, T]))
			lossesVal         = []
			lossesValHighFlow = []
			lossesValLowFlow  = []
			# Cross-validation cycle
			for val in range(20):
				# Calibration
				obs   = flow1_DS.train[val][idxStart:]
				x     = flow1_DS.train[val][0:-1]
				u     = rain1_DS.train[val][0:-1]
				temp  = temp1_DS.train[val][0:-1]
				A = getAarxT(x, N, u, P, temp, T)
				p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
				# Validation over real world data
				obs   = flow_DS.valid[val][idxStart:]
				x     = flow1_DS.valid[val][0:-1]
				u     = rain1_DS.valid[val][0:-1]
				temp  = temp1_DS.valid[val][0:-1]
				X = getXarxT(x, N, u, P, temp, T)
				obs_est = normToRaw(AR(p, X), flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365])
				# Save losses with respect to validation dataset
				lossesVal.append( RMSEloss(obs, obs_est) )
				lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
				lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
			# Save losses with respect to model order
			lossesVal         = np.array(lossesVal)
			lossesValHighFlow = np.array(lossesValHighFlow)
			lossesValLowFlow  = np.array(lossesValLowFlow)
			lossesN[N-1,P-1]     = totalLoss(lossesVal)
			lossesNhigh[N-1,P-1] = totalLoss(lossesValHighFlow)
			lossesNlow[N-1,P-1]  = totalLoss(lossesValLowFlow)
		
	if 0:
		#plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear ARX(N, P, {}) (proper rainfall, proper temperature) results:".format(T))
		plot_linearARXlosses(lossesN, lossesNhigh, lossesNlow, Nref=None, Pref=None)



	# Linear ARX(N, P, T) models with improper X(P+1) and proper temperture T(T): AR(N) + X(P+1) + T(T)
	''' ARX(N, 1):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p[]*u(t) + p[]*u(t+1) + p[]*T(t, t-1, t-2, ..., t-T+1) + p[N]
		ARX(N, P):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p(N-1)*x(t-N+1) +
		           + p[...]*u(t) + p[]*u(t-1)+ ... + p[]*u(t-P+1) + p[]*u(t+1) +
		           + p[]*T(t, t-1, t-2, ..., t-T+1) +
		           + p[-1] 
		u(t) is the standardised input (rain or temperature)
		p is the parameter vector coming out of the lstsq(): N+P+1
	'''
	Nmax = 1
	Pmax = 1
	T    = 1 # T must remain fixed and > 0
	lossesN     = np.zeros((Nmax,Pmax))  # caution: (index) of model is (degree of model -1)
	lossesNhigh = np.zeros((Nmax,Pmax))
	lossesNlow  = np.zeros((Nmax,Pmax))

	for N in range(1, Nmax+1):
		for P in range(1, Pmax+1):
			idxStart = int(np.max([N, P, T]))
			lossesVal         = []
			lossesValHighFlow = []
			lossesValLowFlow  = []
			# Cross-validation cycle
			for val in range(20):
				# Calibration
				obs   = flow1_DS.train[val][idxStart:]
				x     = flow1_DS.train[val][0:-1]
				u     = rain1_DS.train[val][0:-1]
				u_imp = rain1_DS.train[val][idxStart:]
				temp  = temp1_DS.train[val][0:-1]
				A = getAarxImpT(x, N, u, P, u_imp, temp, T)
				p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
				# Validation over real world data
				obs   = flow_DS.valid[val][idxStart:]
				x     = flow1_DS.valid[val][0:-1]
				u     = rain1_DS.valid[val][0:-1]
				u_imp = rain1_DS.valid[val][idxStart:]
				temp  = temp1_DS.valid[val][0:-1]
				X = getXarxImpT(x, N, u, P, u_imp, temp, T)
				obs_est = normToRaw(AR(p, X), flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365])
				# Save losses with respect to validation dataset
				lossesVal.append( RMSEloss(obs, obs_est) )
				lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
				lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
			# Save losses with respect to model order
			lossesVal         = np.array(lossesVal)
			lossesValHighFlow = np.array(lossesValHighFlow)
			lossesValLowFlow  = np.array(lossesValLowFlow)
			lossesN[N-1,P-1]     = totalLoss(lossesVal)
			lossesNhigh[N-1,P-1] = totalLoss(lossesValHighFlow)
			lossesNlow[N-1,P-1]  = totalLoss(lossesValLowFlow)
		
	if 0:
		#plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear ARX(N, P+1, {}) (improper rainfall, proper temperature) results:".format(T))
		plot_linearARXlosses(lossesN, lossesNhigh, lossesNlow, Nref=None, Pref=None)




	# Linear ARX(N, P, T) models with improper X(P+1) and improper temperture T(T+1): AR(N) + X(P+1) + T(T+1)
	''' ARX(N, 1):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p[]*u(t) + p[]*u(t+1) + p[]*T(t, t-1, t-2, ..., t-T+1) + p[]*T(t+1) + p[N]
		ARX(N, P):
		x_est(t+1) = p0*x(t) + p1*x(t-1) + ... + p(N-1)*x(t-N+1) +
		           + p[...]*u(t) + p[]*u(t-1)+ ... + p[]*u(t-P+1) + p[]*u(t+1) +
		           + p[]*T(t, t-1, t-2, ..., t-T+1) + p[]*T(t+1)  +
		           + p[-1] 
		u(t) is the standardised input (rain or temperature)
		p is the parameter vector coming out of the lstsq(): N+P+1
	'''
	Nmax = 1
	Pmax = 1
	T    = 1 # T must remain fixed and > 0
	lossesN     = np.zeros((Nmax,Pmax))  # caution: (index) of model is (degree of model -1)
	lossesNhigh = np.zeros((Nmax,Pmax))
	lossesNlow  = np.zeros((Nmax,Pmax))

	for N in range(1, Nmax+1):
		for P in range(1, Pmax+1):
			idxStart = int(np.max([N, P, T]))
			lossesVal         = []
			lossesValHighFlow = []
			lossesValLowFlow  = []
			# Cross-validation cycle
			for val in range(20):
				# Calibration
				obs      = flow1_DS.train[val][idxStart:]
				x        = flow1_DS.train[val][0:-1]
				u        = rain1_DS.train[val][0:-1]
				u_imp    = rain1_DS.train[val][idxStart:]
				temp     = temp1_DS.train[val][0:-1]
				temp_imp = temp1_DS.train[val][idxStart:]
				A = getAarxImpTimp(x, N, u, P, u_imp, temp, T, temp_imp)
				p = np.array(np.linalg.lstsq(A, obs, rcond=None)[0])
				# Validation over real world data
				obs      = flow_DS.valid[val][idxStart:]
				x        = flow1_DS.valid[val][0:-1]
				u        = rain1_DS.valid[val][0:-1]
				u_imp    = rain1_DS.valid[val][idxStart:]
				temp     = temp1_DS.valid[val][0:-1]
				temp_imp = temp1_DS.valid[val][idxStart:]
				X = getXarxImpTimp(x, N, u, P, u_imp, temp, T, temp_imp)
				obs_est = normToRaw(AR(p, X), flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365])
				# Save losses with respect to validation dataset
				lossesVal.append( RMSEloss(obs, obs_est) )
				lossesValHighFlow.append( RMSElossHigh(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
				lossesValLowFlow.append( RMSElossLow(obs, obs_est, flow_DS.ma[idxStart:365], flow_DS.mv[idxStart:365]) )
			# Save losses with respect to model order
			lossesVal         = np.array(lossesVal)
			lossesValHighFlow = np.array(lossesValHighFlow)
			lossesValLowFlow  = np.array(lossesValLowFlow)
			lossesN[N-1,P-1]     = totalLoss(lossesVal)
			lossesNhigh[N-1,P-1] = totalLoss(lossesValHighFlow)
			lossesNlow[N-1,P-1]  = totalLoss(lossesValLowFlow)
		
	if 0:
		#plot_autocorrelogram(obs - obs_est, order=60)
		print("\nLinear ARX(N, P+1, T+1) (improper rainfall, improper temperature) results:")
		plot_linearARXlosses(lossesN, lossesNhigh, lossesNlow, Nref=None, Pref=None)

	# Clean up some memory
	del flow_DS
	del detr_flow
	del flow1_DS
	del rain_DS 
	del rain1_DS
	del temp_DS
	del r
	del temp1_DS






if PART_11:
	############ PART 1.1: NONLINEAR MODELS : ANN ###################

	# Prepare data (ETL)
	''' Extract data - get data from the source
		Transform data - put data into tensor form
		Load data - put data into an object to make it easily accessible
	'''
	if 1: # Single train/validation cycle
		model_type = "F_"
		model_order = [4]
		include_day = False
		cv_idx = 7
		preproc = 1
		train_dataset = Dataset01(data.copy(),  train=True,
						                        model_type=model_type,
						                        model_order=model_order,
						                        inlude_day_of_year=include_day,
						                        cross_validation_index=cv_idx,
						                        preprocessing=preproc )
		valid_dataset = Dataset01(data.copy(),  train=False,
						                        model_type=model_type,
						                        model_order=model_order,
						                        inlude_day_of_year=include_day,
						                        cross_validation_index=cv_idx,
						                        preprocessing=preproc )
		batch_size = 73 # 5 batrches / year
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
						                             shuffle=True,
						                             num_workers=0 )
		valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
						                             shuffle=True,
						                             num_workers=0 )
		
		# Build model
		n_input = len(train_dataset[0]["input"])
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		net = ANN(n_input).to(device)

		# Train model
		learning_rate = 8.1e-3
		gamma = 0.96 # i think the closer to one, the slower the decay
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
		
		epochs = 100
		train_loss = []
		valid_loss = []
		
		plt.ion()
		fig, ax = plt.subplots()

		for t in range(epochs):
			print(f"\nEpoch {t+1}\n-------------------------------")
			tl = train_loop(train_dataloader, net, device, loss_fn, optimizer)
			train_loss.append(np.sqrt(tl))
			vl = valid_loop(valid_dataloader, net, device, loss_fn)
			valid_loss.append(np.sqrt(vl))
			# Learning rate
			scheduler.step()
			# Plot
			x = np.arange(1, t+1+1)
			ax.clear()
			ax.grid()
			ax.plot( x, np.array(train_loss), "r.-", label="Training")
			ax.plot( x, np.array(valid_loss), "b.-", label="Validation")
			plt.pause(0.1)
			plt.draw()
		
		print("\n\n#########\n#       #\n# Done! #\n#       #\n#########\n")
		plt.ioff()
		plt.show()
	


	if 0: # ITERATION ALONG A MODEL ORDER
		''' '''
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		''' '''
		batch_size = 73 # 5 batches / year
		include_day = False
		preproc = 1
		learning_rate = 8.0e-3
		gamma = 0.96
		epochs = 120
		''' '''
		model_type = "F_"
		i_rain = False
		i_temp = False
		Fmax_, Rmax_, Tmax_ = 15, 0, 0
		orders_to_test = get_modelOrdersToTest(Fmax_, Rmax_, i_rain, Tmax_, i_temp)
		''' '''
		order_cv_loss = []
		''' Plots for training cycle '''
		plt.ion()
		fig_v, axv = plt.subplots()
		''' Plots for Cross-validation (order) cycle '''
		fig_cv, axcv = plt.subplots()
		for order in orders_to_test:
			model_order = order
			print("\n\n\n### Testing model type "+model_type+" with order: ", model_order)
			cv_idx = 0
			train_dataset = Dataset01(data.copy(),  train=True,
						                        model_type=model_type,
						                        model_order=model_order,
						                        inlude_day_of_year=include_day,
						                        cross_validation_index=cv_idx,
						                        preprocessing=preproc )
			valid_dataset = Dataset01(data.copy(),  train=False,
						                        model_type=model_type,
						                        model_order=model_order,
						                        inlude_day_of_year=include_day,
						                        cross_validation_index=cv_idx,
						                        preprocessing=preproc )
			cv_valid_loss = []
			''' cross validation loop: meant to find the best model, not the best network params '''
			for cv_id in range(20):
				print("\n\n Cross validating (CV) index: ", cv_idx)
				train_dataset.setCrossValIdx(cv_id)
				valid_dataset.setCrossValIdx(cv_id)
				train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
															 shuffle=True,
															 num_workers=0 )
				valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
															 shuffle=True,
															 num_workers=0 )
				n_input = len(train_dataset[0]["input"])
				net = ANN(n_input).to(device)
				loss_fn = nn.MSELoss()
				optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
				scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
				train_loss = []
				valid_loss = []
				for t in range(epochs):
					print(f"\nCV {1990+cv_idx}, Epoch {t+1}\n-------------------------------", end="\r")
					tl = train_loop(train_dataloader, net, loss_fn, optimizer, verbose=False)
					vl = valid_loop(valid_dataloader, net, loss_fn, verbose=False)
					# Learning rate
					scheduler.step()
					# Save data (epoch)
					train_loss.append(np.sqrt(tl))
					valid_loss.append(np.sqrt(vl))
					# Plot data
					x = np.arange(1, t+1+1); axv.clear(); axv.grid()
					axv.plot( x, np.array(train_loss), "r.-", label="Training")
					axv.plot( x, np.array(valid_loss), "b.-", label="Validation")
					axv.set_title(f"Epochs cycle for CV year: {1990+cv_id}"); axv.set_xlabel(f"Epochs")
					axv.legend()
					plt.pause(0.1)
					plt.draw()
				print("")
				# Save data (Cross validation)
				cv_valid_loss.append(np.min(valid_loss))
			# Save data about model (order) loss
			order_cv_loss.append(np.median(cv_valid_loss))
			# Plot data
			x = np.arange(1, len(order_cv_loss)+1); axcv.clear(); axcv.grid()
			axcv.plot( x, np.array(order_cv_loss), "r.-", label="Model losses")
			axcv.set_title(f"Model order cycle. Latest in graph: {model_order}"); axcv.set_xlabel(f"Model order")
			plt.pause(0.1)
			plt.draw()
		
		print("\n\n#########\n#       #\n# Done! #\n#       #\n#########\n")
		plt.ioff()
		plt.show()





	






















































### FAILED ATTEMPTS AND USELESS STUFF (NOT TO BE DELETED YET)








print("Program terminated")