import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt




# ###
# STATIONARY STATISTICS
# ###

def printStationaryStats(flow, rain, temp):
	print("STREAMFROW statistics: min, max, range, mean, var, stdev, # days above average statistics per year")
	series = flow
	print(np.min(series))
	print(np.max(series))
	print(np.ptp(series))
	print(np.mean(series))
	print(np.var(series))
	print(np.sqrt(np.var(series)))
	print(np.sum(series > np.mean(series)) / 20)
	print("RAINFALL statistics: min, max, range, mean, var, stdev, # days above average statistics per year")
	series = rain
	print(np.min(series))
	print(np.max(series))
	print(np.ptp(series))
	print(np.mean(series))
	print(np.var(series))
	print(np.sqrt(np.var(series)))
	print(np.sum(series > np.mean(series)) / 20)
	print("TEMPERATURE statistics: min, max, range, mean, var, stdev, # days above average statistics per year")
	series = temp
	print(np.min(series))
	print(np.max(series))
	print(np.ptp(series))
	print(np.mean(series))
	print(np.var(series))
	print(np.sqrt(np.var(series)))





# ###
# CICLOSTATIONARY STATISTICS
# ###

def annualAverage(series):
	T = 365
	if len(series)%T != 0 :
		print("ERROR in annualVariance")
		quit()
	Y = int( len(series) / T )
	series = series.reshape(Y, T)
	out_m = np.mean(series, axis=0) # mean over the years
	out_med = np.median(series, axis=0) # mean over the years
	return out_m

def annualMovingAverage(series, semiwindow=1):
	if (semiwindow < 1):
		semiwindow=1
	w = 2*semiwindow+1
	series = annualAverage(series)
	series = np.concatenate((series[-semiwindow:], series, series[:semiwindow]), axis=None)
	out = np.convolve(series, np.ones(w), 'valid') / (w)
	return out

def annualMovingAverageExtended(series=None, semiwindow=1, annualMA=None, Y=None):
	T = 365
	if annualMA is None:
		if series is None:
			print("ERROR in annualMovingAverageExtended")
			quit()
		Y = int( len(series) / T )
		temp = annualMovingAverage(series, semiwindow)
	else:
		if Y == None:
			print("ERROR in annualMovingAverageExtended: parameter Y missing")
			quit()
		temp = annualMA
	out = [temp for i in range(Y)]
	return np.array(out).reshape(Y*T)

def annualVariance(series):
	T = 365
	if len(series)%T != 0 :
		print("ERROR in annualVariance")
		quit()
	Y = int( len(series) / T )
	series = series.reshape(Y, T)
	out = np.var(series, axis=0) # mean over the years
	return out

def annualMovingVariance(series, semiwindow=1):
	if (semiwindow < 1):
		semiwindow=1
	w = 2*semiwindow+1
	series = annualVariance(series)
	series = np.concatenate((series[-semiwindow:], series, series[:semiwindow]), axis=None)
	out = np.convolve(series, np.ones(w), 'valid') / (w)
	return out

def annualMovingVarianceExtended(series=None, semiwindow=1, annualMV=None):
	T = 365
	if annualMV is None:
		if series is None:
			print("ERROR in annualMovingAverageExtended")
			quit()
		Y = int( len(series) / T )
		temp = annualMovingVariance(series, semiwindow)
	else:
		if Y == None:
			print("ERROR in annualMovingVarianceExtended: parameter Y missing")
			quit()
		temp = annualMV
	out = [temp for i in range(Y)]
	return np.array(out).reshape(Y*T)




# ###
# CORRELOGRAM
# ###

def corr(sig1, sig2=None):
	''' https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation '''
	if sig2 == None:
		sig2 = sig1
	result = np.correlate(sig1, sig2, mode='full')
	return np.array(result[int(result.size/2):])

def autocorr(signal):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    x = signal
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def autocorrelogram(sig1, order=None):
	ac = autocorr(sig1)
	N = len(sig1)
	if order == None:
		order=N-2
	else:
		order=min(order,N-2)
	n =  np.arange(N,N-order-1 -1,-1)
	std_conf_95 =  (n**(-0.5)) * 1.96;    # symmetrical confidence: 0.95 
	std_conf_995 = (n**(-0.5)) * 2.8070; # symmetrical confidence: 0.995
	return (ac[0:order+1], std_conf_95[0:order+1], std_conf_995[0:order+1])




# ###
# DATASETS CREATION
# ###

class Dataset00():
	def __init__(self, series, years, semiwindow=None):
		''' Create a dataset from a time series
			which considers splitting fror cross validation
			Dataset00:
				self.series
				self.ma
				self.mv
				self.train
				self.valid
				self.n_br
				self.br2idx
		'''
		self.series = series
		if semiwindow == None: semiwindow = 7
		self.ma = annualMovingAverageExtended(series=series, semiwindow=semiwindow)
		self.mv = annualMovingVarianceExtended(series=series, semiwindow=semiwindow)
		train = []
		train_ma = []
		train_mv = []
		valid = []
		valid_ma = []
		valid_mv = []
		n_branch = []
		branch2idx = []

		for y in range(1990,2010):
			train.append(series[years != y])
			train_ma.append(annualMovingAverageExtended(series=series[years!=y], semiwindow=semiwindow))
			train_mv.append(annualMovingVarianceExtended(series=series[years!=y], semiwindow=semiwindow))
			valid.append(series[years == y])
			valid_ma.append(annualMovingAverageExtended(series=series[years==y], semiwindow=semiwindow))
			valid_mv.append(annualMovingVarianceExtended(series=series[years==y], semiwindow=semiwindow))
			if y == 1990:
				n_branch.append(1)
				branch2idx.append(-1)
			elif y == 2009:
				n_branch.append(1)
				branch2idx.append(-1)
			else:
				n_branch.append(2)
				branch2idx.append(365*(y-1990))
		
		self.train = np.array(train)
		self.train_ma = np.array(train_ma)
		self.train_mv = np.array(train_mv)
		self.valid = np.array(valid)
		self.valid_ma = np.array(valid_ma)
		self.valid_mv = np.array(valid_mv)
		self.n_br = np.array(n_branch)
		self.br2idx = np.array(branch2idx)


		







# ###
# COST FUNCTIONS
# ###
def RMSEloss(observation, forecast):
	if observation.shape[0] != forecast.shape[0]:
		print("Loss function needs two arrays of same length")
		quit()
	err = (observation - forecast)**2
	return np.sqrt( np.mean(err) )

def R2loss(observation, forecast, ma):
	''' ma is the moving average of observations
	'''
	if ( (observation.shape[0] != forecast.shape[0]) or 
	     (observation.shape[0] != ma.shape[0]) ):
		print("Loss function needs two arrays of same length")
		quit()
	err1 = np.sum( (observation - forecast)**2 )
	err2 = np.sum( (observation - ma)**2 )
	return 1 - err1/err2

def RMSElossHigh(observation, forecast, raw_ma, raw_mv):
	if ( (observation.shape[0] != forecast.shape[0]) or
	     (raw_ma.shape[0] != observation.shape[0]) or
	     (raw_mv.shape[0] != observation.shape[0]) ):
		print("Loss function needs all arrays of same length")
		print("Observation shape: ", observation.shape)
		print("Forecast shape: ", forecast.shape)
		print("MA shape: ", raw_ma.shape)
		print("MV shape: ", raw_mv.shape)
		quit()  
	thresh = raw_ma + 2*np.sqrt(raw_mv)
	obs = observation[observation > thresh]
	frc = forecast[observation > thresh]
	# Avoid output being an empty array
	l = len(obs)
	s = 2.0
	while l < 2:
		s -= 0.05
		thresh = raw_ma + s*np.sqrt(raw_mv)
		obs = observation[observation > thresh]
		frc = forecast[observation > thresh]
		l = len(obs)
	return RMSEloss(obs, frc)

def RMSElossLow(observation, forecast, raw_ma, raw_mv):
	if ( (observation.shape[0] != forecast.shape[0]) or
	     (raw_ma.shape[0] != observation.shape[0]) or
	     (raw_mv.shape[0] != observation.shape[0]) ):
		print("Loss function needs all arrays of same length")
		print("Observation shape: ", observation.shape)
		print("Forecast shape: ", forecast.shape)
		print("MA shape: ", raw_ma.shape)
		print("MV shape: ", raw_mv.shape)
		quit() 
	thresh = raw_ma + 2*np.sqrt(raw_mv)
	obs = observation[observation < thresh]
	frc = forecast[observation < thresh]
	return RMSEloss(obs, frc)

def totalLoss(losses_array):
	if len(losses_array) != 20:
		print("The total loss must be computed from the 20 cross-validation losses.")
		quit()
	return np.median(losses_array)












# ###
# LINEAR MODELS
# ###

def getA(x, N):
	A = x[N-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[N-1-i:-i]] )
	A = np.vstack( [A, np.ones(len(x[N-1:]))] ).T
	return A

def getX(x, N):
	X = x[N-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[N-1-i:-i]] )
	if N == 1:
		X = np.transpose([X])
	else:
		X = X.T
	return X

def getApoli(x, N):
	A = x[N-1:]**N
	for i in range(1,N):
		A = np.vstack( [A, x[N-1-i:-i]**(N-i)] )
	A = np.vstack( [A, np.ones(len(x[N-1:]))] ).T
	return A

def getXpoli(x, N):
	X = x[N-1:]**N
	for i in range(1,N):
		X = np.vstack( [X, x[N-1-i:-i]**(N-i)] )
	if N == 1:
		X = np.transpose([X])
	else:
		X = X.T
	return X

def getAarx(x, N, u, P):
	idxStart = int(np.max([N, P]))
	A = x[idxStart-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[idxStart-1-i:-i]] )
	A = np.vstack( [A, u[idxStart-1:]] )
	for i in range(1,P):
		A = np.vstack( [A, u[idxStart-1-i:-i]] )
	A = np.vstack( [A, np.ones(len(x[idxStart-1:]))] ).T
	return A

def getXarx(x, N, u, P):
	idxStart = int(np.max([N, P]))
	X = x[idxStart-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[idxStart-1-i:-i]] )
	X = np.vstack( [X, u[idxStart-1:]] )
	for i in range(1,P):
		X = np.vstack( [X, u[idxStart-1-i:-i]] )
	X = X.T
	return X

def getAarxImp(x, N, u, P, u_imp):
	''' u_imp has same exact shape and indices of the observations vector '''
	idxStart = int(np.max([N, P]))
	A = x[idxStart-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[idxStart-1-i:-i]] )
	A = np.vstack( [A, u[idxStart-1:]] )
	for i in range(1,P):
		A = np.vstack( [A, u[idxStart-1-i:-i]] )
	A = np.vstack( [A, u_imp] )
	A = np.vstack( [A, np.ones(len(x[idxStart-1:]))] ).T
	return A

def getXarxImp(x, N, u, P, u_imp):
	''' u_imp has same exact shape and indices of the observations vector '''
	idxStart = int(np.max([N, P]))
	X = x[idxStart-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[idxStart-1-i:-i]] )
	X = np.vstack( [X, u[idxStart-1:]] )
	for i in range(1,P):
		X = np.vstack( [X, u[idxStart-1-i:-i]] )
	X = np.vstack( [X, u_imp] )
	X = X.T
	return X

def getAarxT(x, N, u, P, temp, T):
	''' u_imp has same exact shape and indices of the observations vector '''
	# x
	idxStart = int(np.max([N, P, T]))
	A = x[idxStart-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[idxStart-1-i:-i]] )
	# u
	A = np.vstack( [A, u[idxStart-1:]] )
	for i in range(1,P):
		A = np.vstack( [A, u[idxStart-1-i:-i]] )
	# temp
	A = np.vstack( [A, temp[idxStart-1:]] )
	for i in range(1,T):
		A = np.vstack( [A, temp[idxStart-1-i:-i]] )
	# ones
	A = np.vstack( [A, np.ones(len(x[idxStart-1:]))] ).T
	return A

def getXarxT(x, N, u, P, temp, T):
	''' u_imp has same exact shape and indices of the observations vector '''
	idxStart = int(np.max([N, P, T]))
	# x
	X = x[idxStart-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[idxStart-1-i:-i]] )
	# u
	X = np.vstack( [X, u[idxStart-1:]] )
	for i in range(1,P):
		X = np.vstack( [X, u[idxStart-1-i:-i]] )
	# temp
	X = np.vstack( [X, temp[idxStart-1:]] )
	for i in range(1,T):
		X = np.vstack( [X, temp[idxStart-1-i:-i]] )
	# transpose
	X = X.T
	return X

def getAarxImpT(x, N, u, P, u_imp, temp, T):
	''' u_imp has same exact shape and indices of the observations vector '''
	# x
	idxStart = int(np.max([N, P, T]))
	A = x[idxStart-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[idxStart-1-i:-i]] )
	# u
	A = np.vstack( [A, u[idxStart-1:]] )
	for i in range(1,P):
		A = np.vstack( [A, u[idxStart-1-i:-i]] )
	# u_imp
	A = np.vstack( [A, u_imp] )
	# temp
	A = np.vstack( [A, temp[idxStart-1:]] )
	for i in range(1,T):
		A = np.vstack( [A, temp[idxStart-1-i:-i]] )
	# ones
	A = np.vstack( [A, np.ones(len(x[idxStart-1:]))] ).T
	return A

def getXarxImpT(x, N, u, P, u_imp, temp, T):
	''' u_imp has same exact shape and indices of the observations vector '''
	idxStart = int(np.max([N, P, T]))
	# x
	X = x[idxStart-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[idxStart-1-i:-i]] )
	# u
	X = np.vstack( [X, u[idxStart-1:]] )
	for i in range(1,P):
		X = np.vstack( [X, u[idxStart-1-i:-i]] )
	# u_imp
	X = np.vstack( [X, u_imp] )
	# temp
	X = np.vstack( [X, temp[idxStart-1:]] )
	for i in range(1,T):
		X = np.vstack( [X, temp[idxStart-1-i:-i]] )
	# transpose
	X = X.T
	return X

def getAarxImpTimp(x, N, u, P, u_imp, temp, T, temp_imp):
	''' u_imp and temp_imp have same exact shape and indices of the observations vector '''
	# x
	idxStart = int(np.max([N, P, T]))
	A = x[idxStart-1:]
	for i in range(1,N):
		A = np.vstack( [A, x[idxStart-1-i:-i]] )
	# u
	A = np.vstack( [A, u[idxStart-1:]] )
	for i in range(1,P):
		A = np.vstack( [A, u[idxStart-1-i:-i]] )
	# u_imp
	A = np.vstack( [A, u_imp] )
	# temp
	A = np.vstack( [A, temp[idxStart-1:]] )
	for i in range(1,T):
		A = np.vstack( [A, temp[idxStart-1-i:-i]] )
	# temp_imp
	A = np.vstack( [A, temp_imp] )
	# ones
	A = np.vstack( [A, np.ones(len(x[idxStart-1:]))] ).T
	return A

def getXarxImpTimp(x, N, u, P, u_imp, temp, T, temp_imp):
	''' u_imp and temp_imp have same exact shape and indices of the observations vector '''
	idxStart = int(np.max([N, P, T]))
	# x
	X = x[idxStart-1:]
	for i in range(1,N):
		X = np.vstack( [X, x[idxStart-1-i:-i]] )
	# u
	X = np.vstack( [X, u[idxStart-1:]] )
	for i in range(1,P):
		X = np.vstack( [X, u[idxStart-1-i:-i]] )
	# u_imp
	X = np.vstack( [X, u_imp] )
	# temp
	X = np.vstack( [X, temp[idxStart-1:]] )
	for i in range(1,T):
		X = np.vstack( [X, temp[idxStart-1-i:-i]] )
	# temp_imp
	X = np.vstack( [X, temp_imp] )
	# transpose
	X = X.T
	return X

def normToRaw(datum, ma, mv, day_idx=None):
	'''
	This function takes in the normalised value of the
	timeseries (and other parameters) and outputs
	the value as it would be in the raw dataset.
	It is the inverse of the normalization - detrending - standardisation
	process
	'''
	if day_idx == None:
		datum = datum * np.sqrt(mv) + ma
	else:
		if datum.size > 1:
			quit()
		datum = datum * np.sqrt(mv[day_idx]) + ma[day_idx]
	return datum

def lognormToRaw(datum, ma, mv, day_idx=None):
	'''
	This function takes in the normalised value of the
	timeseries (and other parameters) and outputs
	the value as it would be in the raw dataset.
	It is the inverse of the normalization - detrending - standardisation
	process
	'''
	if day_idx == None:
		datum = datum * np.sqrt(mv) + ma
	else:
		if datum.size > 1:
			quit()
		datum = datum * np.sqrt(mv[day_idx]) + ma[day_idx]
	return np.exp(datum)

def AR(p, X):
	'''
	out = p[0]*x[0] + p[1]*x[1] + ... + p[N]*1
	t+1       t          t-1      ...      t-N
	Input dimensions:
		AR(N) for 1 input vector: [[x0 x1 x2 x3 ... xN]] row vector (1,N)
		AR(1) for M input vectors:  [x00   column vector (M,1)
									 x01
									 x02
									 ...
									 x0M]
		AR(N) for M input vectors: (MxN)
		                      [[x0 x1 x2 x3 ... xN],  0
		                       [x0 x1 x2 x3 ... xN],  1
		                                ...          ...
		                       [x0 x1 x2 x3 ... xN]]  M
		x.dot(p) means x * p (row times column)

	'''
	if X.size == 1:
		if p.size != 2:
			print("Error in AR model input dimensions.")
			quit()
		return p[0]*X + p[1]
	else:
		X = np.append(X, np.ones((X.shape[0],1)), axis=1)
		out = X.dot(p)
		return out



# ###
# title
# ###




