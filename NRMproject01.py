# python3.8
import numpy as np
import scipy
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
import torch

from NRMproject_utils0 import *
from NRMproject_utils1_ANN import *
from NRMproject_plots import *
from NRMproject_utils2_trees import *

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
flow  = data[:,4] # m3/s
temp  = data[:,5] # °C

### RAW DATA VISUALIZATION
if 0 : plot_rawData(flow, rain, temp, year)

### STATIONARY STATISTICS
if 0 : printStationaryStats(flow, rain, temp)

### PROGRAM COMPARTMENTS
PART_10 = 0
PART_11 = 0
PART_12 = 1





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
	
	if 0: # Single train/validation cycle
		''' model_type: save input type as indexes.
			Allowed:
			0: ""
			0: "F_"
			1: "F_R_PRO"
			2: "F_R_IMP"
			3: "F_T_PRO"
			4: "F_T_IMP"
			5: "F_R_PRO_T_PRO"
			6: "F_R_IMP_T_PRO"
			7: "F_R_IMP_T_IMP"
		'''
		model_type = "F_R_PRO_T_PRO"
		model_order = [3, 4, 5]
		include_day = False
		cv_idx = 17
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
		learning_rate = net.param_lr_gamma["deepNonlinear"][0]
		gamma         = net.param_lr_gamma["deepNonlinear"][1] # the closer to one, the slower the decay
		
		#print(net.linear[0].weight, net.linear[0].bias)

		# Train model
		loss_fn = nn.MSELoss()
		optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
		
		epochs = 100
		train_loss = []
		valid_loss = []
		valid_loss_H = []
		
		plt.ion()
		fig, axv = plt.subplots()

		for t in range(epochs):
			print(f"\nEpoch {t+1}\n-------------------------------")
			tl = train_loop(train_dataloader, net, device, loss_fn, optimizer)
			train_loss.append(np.sqrt(tl))
			vl, vl_H = valid_loop(valid_dataloader, net, device, loss_fn, verbose=False, high_flow_loss=True)
			valid_loss.append(np.sqrt(vl))
			valid_loss_H.append(np.sqrt(vl_H))
			# Learning rate
			scheduler.step()
			# Plot data
			if (t % 10 == 0) or (t>=epochs-10):
				x = np.arange(1, t+1+1); axv.clear(); axv.grid()
				axv.plot( x, np.array(train_loss), "r.-", label="Training")
				axv.plot( x, np.array(valid_loss), "b.-", label="Validation")
				axv.plot( x, np.array(valid_loss_H), "b.--", linewidth=0.9, label="Validation high flow")
				axv.set_title(f"Epochs cycle for CV year: {1990+cv_idx}| Model order: {model_order}")
				axv.set_xlabel(f"Epochs")
				axv.legend()
				plt.pause(0.001)
				plt.draw()
		
		print("\n\n#########\n#       #\n# Done! #\n#       #\n#########\n")
		plt.ioff()
		plt.show()
		
		# Scatterplot of labels-predictions and timeseries prediction for validation
		plot_NNresults(valid_dataset, net, cv_idx)

	


	if 0: # ITERATION ALONG A MODEL ORDER
		''' '''
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		''' '''
		batch_size = 73 # 5 batches / year
		preproc = 1
		epochs = 130
		learning_rate = ANN(1).param_lr_gamma["deepNonlinear"][0]
		gamma         = ANN(1).param_lr_gamma["deepNonlinear"][1]
		''' '''
		model_type = "F_R_PRO_T_PRO"
		include_day = False
		i_rain = False
		i_temp = False
		Fmax_, Rmax_, Tmax_ = 5, 6, 6
		orders_to_test = get_modelOrdersToTest(Fmax_, Rmax_, i_rain, Tmax_, i_temp)
		#orders_to_test = [[5, 5, 10]]
		''' '''
		order_cv_loss = []
		order_cv_loss_H = []
		''' Plots for training cycle '''
		plt.ion()
		fig_v, axv = plt.subplots()
		''' Plots for Cross-validation (order) cycle '''
		fig_cv, axcv = plt.subplots()
		for order in orders_to_test:
			model_order = order
			print("\n\n\n### Testing model type "+model_type+" with order: ", model_order)
			cv_valid_loss = []
			cv_valid_loss_H = []
			''' cross validation loop: meant to find the best model, not the best network params '''
			for cv_id in range(20):
				print("Cross validating (CV) index: ", cv_id)
				train_dataset = Dataset01(data.copy(),  train=True,
								                        model_type=model_type,
								                        model_order=model_order,
								                        inlude_day_of_year=include_day,
								                        cross_validation_index=cv_id,
								                        preprocessing=preproc )
				valid_dataset = Dataset01(data.copy(),  train=False,
								                        model_type=model_type,
								                        model_order=model_order,
								                        inlude_day_of_year=include_day,
								                        cross_validation_index=cv_id,
								                        preprocessing=preproc )
				train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
															 shuffle=True,
															 num_workers=0 )
				valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
															 shuffle=False,
															 num_workers=0 )
				n_input = len(train_dataset[0]["input"])
				net = ANN(n_input).to(device)
				loss_fn = nn.MSELoss()
				optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
				scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=False)
				train_loss = []
				valid_loss = []
				valid_loss_H = []
				for t in range(epochs):
					print(f"CV {1990+cv_id}, Epoch {t+1} ", end="\r")
					tl = train_loop(train_dataloader, net, device, loss_fn, optimizer, verbose=False)
					vl, vl_H = valid_loop(valid_dataloader, net, device, loss_fn, verbose=False, high_flow_loss=True)
					# Learning rate
					scheduler.step()
					# Save data (epoch)
					train_loss.append(np.sqrt(tl))
					valid_loss.append(np.sqrt(vl))
					valid_loss_H.append(np.sqrt(vl_H))
					# Plot data
					if (t == 20) or (t % 40 == 0) or (t>=epochs-6):
						x = np.arange(1, t+1+1); axv.clear(); axv.grid()
						axv.plot( x, np.array(train_loss), "r.-", label="Training")
						axv.plot( x, np.array(valid_loss), "b.-", label="Validation")
						axv.plot( x, np.array(valid_loss_H), "b.--", linewidth=0.9, label="Validation high flow")
						axv.set_title(f"Epochs cycle for CV year: {1990+cv_id}| Model order: {order}"); axv.set_xlabel(f"Epochs")
						axv.legend()
						plt.pause(0.01)
						plt.draw()
				print("")
				# Save data (Cross validation)
				cv_valid_loss.append(np.min(valid_loss[-30:]))
				cv_valid_loss_H.append(np.min(valid_loss_H[-30:]))
			# Save data about model (order) loss
			order_cv_loss.append(np.median(cv_valid_loss))
			order_cv_loss_H.append(np.median(cv_valid_loss_H))
			# Plot data
			x = np.arange(1, len(order_cv_loss)+1); axcv.clear(); axcv.grid()
			axcv.plot( x, np.array(order_cv_loss), "r.-", label="All flows")
			axcv.plot( x, np.array(order_cv_loss_H), "r.--", label="High flows only")
			axcv.set_title("Model "+model_type+f" order cycle. Latest in graph: {model_order}"); axcv.set_xlabel(f"Model order")
			axcv.legend()
			plt.pause(0.01)
			plt.draw()
		plt.ioff()
		plt.show()
		# Print data:
		print("\n\n#########\n#       #\n# Done! #\n#       #\n#########\n")
		for i in range(len(order_cv_loss)):
			print(orders_to_test[i], ": ", order_cv_loss[i], " , ", order_cv_loss_H[i])





























if PART_12:
	from sklearn.tree import DecisionTreeRegressor
	'''
	class sklearn.tree.DecisionTreeRegressor( criterion='squared_error', splitter='best', max_depth=None,
	                                          min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	                                          max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
	                                          ccp_alpha=0.0)
	'''
	if 0:
		# Parameters
		criterion="friedman_mse" # "squared_error"
		splitter ="best" # random
		max_depth = 15
		min_samples_split=2 # The minimum number of samples required to split an internal node
		min_samples_leaf =1 # minimum samples required per leaf
		max_leaf_nodes=None


		# Get data
		model_type  = "F_R_PRO_T_PRO"
		model_order = [4, 3, 3]
		include_day = False
		cv_id   = 17
		preproc = 0

		Xt, Yt = getTreesData(data.copy(),
							train=True,
	                        model_type=model_type,
	                        model_order=model_order,
	                        inlude_day_of_year=include_day,
	                        cross_validation_index=cv_id,
	                        preprocessing=preproc,
	                        HL=False )
		Xv, Xv_L, Xv_H, Yv, Yv_L, Yv_H = getTreesData(data.copy(),  
														train=False,
								                        model_type=model_type,
								                        model_order=model_order,
								                        inlude_day_of_year=include_day,
								                        cross_validation_index=cv_id,
								                        preprocessing=preproc,
								                        HL=True )

		# Fit regression model
		regr_1 = DecisionTreeRegressor(criterion=criterion,
			                           splitter=splitter,
			                           max_depth=max_depth,
		                               min_samples_split=min_samples_split, 
		                               min_samples_leaf=min_samples_leaf,
		                               max_leaf_nodes=max_leaf_nodes)
		regr_1.fit(Xt, Yt)

		# Predict
		Y_ = regr_1.predict(Xv)
		
		print(treeRMSE(Y_, Yv))

		plt.plot(Yvalid)
		plt.plot(Y_)
		plt.show()


	if 0: # Trees
		# Parameters
		criterion="friedman_mse" # "squared_error"
		splitter ="best" # random
		max_depth = 10
		min_samples_split=4 # The minimum number of samples required to split an internal node
		min_samples_leaf =4 # minimum samples required per leaf
		max_leaf_nodes=None

		model_type  = "F_R_PRO"
		include_day = False
		orders_to_test = get_modelOrdersToTest(4, 4, False, 0, False)
		preproc = 0

		plt.ion()
		fig_v, axv = plt.subplots()


		loss, lossL, lossH = [], [], []
		for model_order in orders_to_test:
			s, sL, sH = 0, 0, 0
			for cv_id in range(20):
				Xt, Yt = getTreesData(data.copy(),
									train=True,
			                        model_type=model_type,
			                        model_order=model_order,
			                        inlude_day_of_year=include_day,
			                        cross_validation_index=cv_id,
			                        preprocessing=preproc,
			                        HL=False )
				Xv, Xv_L, Xv_H, Yv, Yv_L, Yv_H = getTreesData(data.copy(),  
																train=False,
										                        model_type=model_type,
										                        model_order=model_order,
										                        inlude_day_of_year=include_day,
										                        cross_validation_index=cv_id,
										                        preprocessing=preproc,
										                        HL=True )
				numTrials = 30
				score, scoreL, scoreH = 0, 0, 0
				for nt in range(numTrials):
					# Fit regression model
					regr_1 = DecisionTreeRegressor(criterion=criterion,
						                           splitter=splitter,
						                           max_depth=max_depth,
					                               min_samples_split=min_samples_split, 
					                               min_samples_leaf=min_samples_leaf,
					                               max_leaf_nodes=max_leaf_nodes)
					regr_1.fit(Xt, Yt)
					# Predict
					Y_ = regr_1.predict(Xv);     score += treeRMSE(Y_, Yv)/numTrials
					Y_L = regr_1.predict(Xv_L);  scoreL+= treeRMSE(Y_L, Yv_L)/numTrials
					Y_H = regr_1.predict(Xv_H);  scoreH+= treeRMSE(Y_H, Yv_H)/numTrials

				s += score/20
				sL += scoreL/20
				sH += scoreH/20
			loss.append(s)
			lossL.append(sL)
			lossH.append(sH)
			# plot and print
			print(model_order, s, sL, sH)
			axv.clear()
			x = range(1, len(loss)+1)
			axv.plot(x, loss, "b.-")
			axv.plot(x, lossL, "g.-")
			axv.plot(x, lossH, "r.-")
			axv.grid()
			plt.pause(0.1)
			plt.draw()


		plt.ioff()
		plt.figure()
		x = range(1, len(loss)+1)
		plt.plot(x, loss, "b.-")
		plt.plot(x, lossL, "g.-")
		plt.plot(x, lossH, "r.-")
		plt.grid()
		plt.show()



	if 0: # Bagged trees
		# Parameters
		criterion="friedman_mse" # "squared_error"
		splitter ="best" # random
		max_depth = 10
		min_samples_split=4 # The minimum number of samples required to split an internal node
		min_samples_leaf =4 # minimum samples required per leaf
		max_leaf_nodes=None

		model_type  = "F_R_PRO"
		include_day = False
		orders_to_test = get_modelOrdersToTest(3, 3, False, 0, False)
		orders_to_test = [[2, 1]]
		preproc = 0

		plt.ion()
		fig_v, axv = plt.subplots()


		loss, lossL, lossH = [], [], []
		for model_order in orders_to_test:
			s, sL, sH = 0, 0, 0
			for cv_id in range(20):
				Xt, Yt = getTreesData(data.copy(),
									train=True,
			                        model_type=model_type,
			                        model_order=model_order,
			                        inlude_day_of_year=include_day,
			                        cross_validation_index=cv_id,
			                        preprocessing=preproc,
			                        HL=False )
				Xv, Xv_L, Xv_H, Yv, Yv_L, Yv_H = getTreesData(data.copy(),  
																train=False,
										                        model_type=model_type,
										                        model_order=model_order,
										                        inlude_day_of_year=include_day,
										                        cross_validation_index=cv_id,
										                        preprocessing=preproc,
										                        HL=True )
				numTrials = 10
				score, scoreL, scoreH = 0, 0, 0
				for nt in range(numTrials):
					# Fit regression model
					bagged = baggedTree(numTrees=200,criterion=criterion,
						                            splitter=splitter,
						                            max_depth=max_depth,
					                                min_samples_split=min_samples_split, 
					                                min_samples_leaf=min_samples_leaf,
					                                max_leaf_nodes=max_leaf_nodes)
					bagged.fit(Xt, Yt)
					# Predict
					Y_ = bagged.predict(Xv);     score += treeRMSE(Y_, Yv)/numTrials
					Y_L = bagged.predict(Xv_L);  scoreL+= treeRMSE(Y_L, Yv_L)/numTrials
					Y_H = bagged.predict(Xv_H);  scoreH+= treeRMSE(Y_H, Yv_H)/numTrials

				s += score/20
				sL += scoreL/20
				sH += scoreH/20
			loss.append(s)
			lossL.append(sL)
			lossH.append(sH)
			# plot and print
			print(model_order, s, sL, sH)
			axv.clear()
			x = range(1, len(loss)+1)
			axv.plot(x, loss, "b.-")
			axv.plot(x, lossL, "g.-")
			axv.plot(x, lossH, "r.-")
			axv.grid()
			plt.pause(0.1)
			plt.draw()


		plt.ioff()
		plt.figure()
		x = range(1, len(loss)+1)
		plt.plot(x, loss, "b.-")
		plt.plot(x, lossL, "g.-")
		plt.plot(x, lossH, "r.-")
		plt.grid()
		plt.show()



	if 0: # Random forests
		from sklearn.ensemble import RandomForestRegressor
		# Parameters
		criterion="friedman_mse" # "squared_error"
		splitter ="best" # random
		max_depth = 10
		min_samples_split=4 # The minimum number of samples required to split an internal node
		min_samples_leaf =4 # minimum samples required per leaf
		max_leaf_nodes=None

		model_type  = "F_R_PRO"
		model_order = [2, 1]
		include_day = False
		preproc = 0

		plt.ion()
		fig_v, axv = plt.subplots()


		loss, lossL, lossH = [], [], []
		# cycle over number of trees
		NT = [3, 5, 10, 15, 20]
		for N in NT:
			s, sL, sH = 0, 0, 0
			for cv_id in range(20):
				Xt, Yt = getTreesData(data.copy(),
									train=True,
			                        model_type=model_type,
			                        model_order=model_order,
			                        inlude_day_of_year=include_day,
			                        cross_validation_index=cv_id,
			                        preprocessing=preproc,
			                        HL=False )
				Xv, Xv_L, Xv_H, Yv, Yv_L, Yv_H = getTreesData(data.copy(),  
																train=False,
										                        model_type=model_type,
										                        model_order=model_order,
										                        inlude_day_of_year=include_day,
										                        cross_validation_index=cv_id,
										                        preprocessing=preproc,
										                        HL=True )
				numTrials = 15
				score, scoreL, scoreH = 0, 0, 0
				for nt in range(numTrials):
					# Fit regression model
					forest = RandomForestRegressor(n_estimators=N,
						                            max_samples=0.70,
						                            criterion=criterion,
						                            max_depth=max_depth,
					                                min_samples_split=min_samples_split, 
					                                min_samples_leaf=min_samples_leaf,
					                                max_leaf_nodes=max_leaf_nodes)
					forest.fit(Xt, Yt)
					# Predict
					Y_ = forest.predict(Xv);     score += treeRMSE(Y_, Yv)/numTrials
					Y_L = forest.predict(Xv_L);  scoreL+= treeRMSE(Y_L, Yv_L)/numTrials
					Y_H = forest.predict(Xv_H);  scoreH+= treeRMSE(Y_H, Yv_H)/numTrials

				s += score/20
				sL += scoreL/20
				sH += scoreH/20
			loss.append(s)
			lossL.append(sL)
			lossH.append(sH)
			# plot and print
			print(N, s, sL, sH)
			axv.clear()
			x = range(1, len(loss)+1)
			axv.plot(x, loss, "b.-")
			axv.plot(x, lossL, "g.-")
			axv.plot(x, lossH, "r.-")
			axv.grid()
			plt.pause(0.1)
			plt.draw()


		plt.ioff()
		plt.figure()
		plt.plot(NT, loss, "b.-")
		plt.plot(NT, lossL, "g.-")
		plt.plot(NT, lossH, "r.-")
		plt.grid()
		plt.show()





















































### FAILED ATTEMPTS AND USELESS STUFF (NOT TO BE DELETED YET)








print("Program terminated")
