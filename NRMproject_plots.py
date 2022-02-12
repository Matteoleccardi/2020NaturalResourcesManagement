import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from NRMproject_utils0 import *
from NRMproject_utils1_ANN import *

def getRGBA():
	# Preferred colormap
	cmap = matplotlib.cm.get_cmap('Spectral')
	rgba = cmap( np.arange(0,20)/20 )
	return rgba

def getFC():
	return (.35, .35, .35)

def plot_rawData(flow, rain, temp, year):
	fig = plt.figure()
	spec = fig.add_gridspec(nrows=2, ncols=3)
	rgba = getRGBA()
	# Plot 20 Years of streamflow
	dates = np.arange(str(int(year[0]))+'-01-01',
		              str(int(year[-1])+1)+'-01-01',
		              dtype='datetime64[D]')
	dates = dates[ dates != np.datetime64('1992-02-29') ]
	dates = dates[ dates != np.datetime64('1996-02-29') ]
	dates = dates[ dates != np.datetime64('2000-02-29') ]
	dates = dates[ dates != np.datetime64('2004-02-29') ]
	dates = dates[ dates != np.datetime64('2008-02-29') ]
	ax0 = fig.add_subplot(spec[0, :])
	for i in range(0,20):
		ax0.plot(dates[365*i:365*i+365], flow[year==(1990+i)], alpha=1, c=rgba[i], zorder=3)
	for i in range(0,20):
		ax0.plot([dates[365*i], dates[365*i]], [0, 400], '--', linewidth=0.7, alpha=0.5, c=[0.7, 0.7, 0.7], zorder=2)
	ax0.plot([dates[-1], dates[-1]], [0, 400], '--', linewidth=0.7, alpha=0.5, c=[0.7, 0.7, 0.7], zorder=2)
	ax0.grid(linestyle='--', linewidth=0.5)
	ax0.set_title("Streamflow complete time series")
	ax0.set_ylabel("Daily mean $m^{3}/s$")
	ax0.set_xlabel("")
	ax0.set_facecolor(getFC())
	# Plot color-changing yearly data
	x = np.arange(1, 365+1)
	# Flow
	series = flow
	ax1 = fig.add_subplot(spec[1, 0])
	i=0
	for y in range(1990,2010):
		ax1.plot(x, series[year==y], alpha=0.8, c=rgba[i])
		i += 1
	ax1.grid(linestyle='--', linewidth=0.5)
	ax1.set_title("Stacked annual streamflow")
	ax1.set_ylabel("$m^{3}/s$")
	ax1.set_facecolor(getFC())
	# Rain
	series = rain
	ax2 = fig.add_subplot(spec[1, 1])
	i=0
	for y in range(1990,2010):
		ax2.plot(x, series[year==y], alpha=0.8, c=rgba[i])
		i += 1
	ax2.grid(linestyle='--', linewidth=0.5)
	ax2.set_title("Stacked annual rain")
	ax2.set_ylabel("mm")
	ax2.set_xlabel("Days since Jan. 1 for each year")
	ax2.set_facecolor(getFC())
	# Temp
	series = temp
	ax3 = fig.add_subplot(spec[1, 2])
	i=0
	for y in range(1990,2010):
		ax3.plot(x, series[year==y], alpha=0.8, c=rgba[i])
		i += 1
	ax3.grid(linestyle='--', linewidth=0.5)
	ax3.set_title("Stacked annual temperature")
	ax3.set_ylabel("°C")
	ax3.set_facecolor(getFC())
	# Show plot
	plt.show()







def plot_autocorrelogram(sig1, order=None):
	rgba = getRGBA()
	FC = getFC()
	(ac_1, ac95_1, ac995_1) = autocorrelogram(sig1, order=order)
	fig = plt.figure()
	ax2 = fig.add_subplot()
	ax2.plot( ac_1, '.-', c=rgba[17], label="Autocorr")
	ax2.fill_between(range(0,order+1), ac95_1, -ac95_1, color=rgba[2], alpha = 0.2, zorder=3, label="95% confidence")
	ax2.fill_between(range(0,order+1), ac995_1, -ac995_1, color=rgba[7], alpha = 0.2, zorder=2, label="99.5% confidence")
	ax2.grid(linestyle='--', linewidth=0.5)
	ax2.set_title("Autocorrelogram")
	ax2.set_ylabel("Correlation")
	ax2.set_xlabel("Lag")
	ax2.legend()
	ax2.set_facecolor(getFC())
	plt.show()





def plot_seriesDetrending(flow_DS, flow1_DS, day_y, obj=""):
	rgba = getRGBA()
	FC = getFC()
	fig = plt.figure()
	spec = fig.add_gridspec(nrows=2, ncols=2)
	if obj == "": obj = "series"
	# flow_DS plot with ma and mv
	ax0 = fig.add_subplot(spec[0, 0])
	ax0.plot(day_y,flow_DS.series, '.', c=rgba[17])
	ax0.plot(range(1,366), flow_DS.ma[0:365],  '--', c=rgba[0], alpha=0.9, linewidth=2, label="Moving Avg.", zorder=3)
	ax0.fill_between(range(1,366), flow_DS.ma[0:365] + 2*np.sqrt(flow_DS.mv[0:365]), flow_DS.ma[0:365] - 2*np.sqrt(flow_DS.mv[0:365]), color=rgba[6], alpha = 0.3, label="95% C.I.", zorder=2)
	ax0.grid(linestyle='--', linewidth=0.5)
	ax0.set_title("Yearly "+obj+" over 20 years")
	ax0.set_ylabel("Units")
	ax0.set_xlabel("")
	ax0.set_facecolor(getFC())
	ax0.legend()
	# flow1_DS plot with ma and mv
	ax1 = fig.add_subplot(spec[0, 1])
	ax1.plot(day_y,flow1_DS.series, '.', c=rgba[17])
	ax1.plot(range(1,366), flow1_DS.ma[0:365],  '--', c=rgba[0], alpha=0.9, linewidth=2, zorder=3)
	ax1.fill_between(range(1,366), flow1_DS.ma[0:365] + np.sqrt(flow1_DS.mv[0:365]), flow1_DS.ma[0:365] - np.sqrt(flow1_DS.mv[0:365]), color=rgba[6], alpha = 0.3, zorder=2)
	ax1.grid(linestyle='--', linewidth=0.5)
	ax1.set_title("Detrended and standardised")
	ax1.set_ylabel("Normalised units")
	ax1.set_xlabel("")
	ax1.set_facecolor(getFC())
	# Correlogram of standardised flow
	order = 60
	(ac_1, ac95_1, ac995_1) = autocorrelogram(flow1_DS.series, order=order)
	ax2 = fig.add_subplot(spec[1, 0])
	ax2.plot( ac_1, '.-', c=rgba[17], label="Autocorr")
	ax2.fill_between(range(0,order+1), ac95_1, -ac95_1, color=rgba[2], alpha = 0.2, zorder=3, label="95% confidence")
	ax2.fill_between(range(0,order+1), ac995_1, -ac995_1, color=rgba[7], alpha = 0.2, zorder=2, label="99.5% confidence")
	ax2.grid(linestyle='--', linewidth=0.5)
	ax2.set_title("Standardised "+obj+" autocorrelation")
	ax2.set_ylabel("Correlation")
	ax2.set_xlabel("Lag")
	ax2.legend()
	ax2.set_facecolor(getFC())
	# Histogram of standardised flow
	ax3 = fig.add_subplot(spec[1, 1])
	ax3.hist( flow1_DS.series, color=rgba[17], bins=40, alpha=0.9, density=True, log=False)
	ax3.grid(linestyle='--', linewidth=0.5)
	ax3.set_title("Standardised "+obj+" distribution")
	ax3.set_ylabel("")
	ax3.set_xlabel("Normalised units")
	ax3.set_facecolor(getFC())
	# Show plot
	plt.show()




def plot_linearARlosses(loss, lossH, lossL, add_title=""):
	rgba = getRGBA()
	FC = getFC()
	fig = plt.figure()
	ax = fig.add_subplot()
	# plot losses
	x = np.arange(1, len(loss)+1)
	ax.plot(x,loss, '.-', c=rgba[16], label="Global loss $J_{FINAL}$")
	ax.plot(x,lossH, '.-', c=rgba[1], label="$J_{FINAL}$ for high flows")
	ax.plot(x,lossL, '.-', c=rgba[19], label="$J_{FINAL}$ for medium/low flows")
	# plot lowest points
	i = np.argsort(loss)[0]
	ax.plot(i+1,loss[i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossH)[0]
	ax.plot(i+1,lossH[i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossL)[0]
	ax.plot(i+1,lossL[i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	# Plot figure
	ax.grid(linestyle='--', linewidth=0.5)
	ax.set_title("Total losses ($J_{FINAL}$) " + add_title)
	ax.set_ylabel("Units")
	ax.set_xlabel("model order N")
	ax.set_facecolor(getFC())
	ax.legend()
	# PRINT BEST INDICES OVERALL AND Jtotal VALUES
	i = np.argwhere(loss == np.min(loss)).flatten()
	print("Best AR model globally: AR("+str(i[0]+1)+"), with Jtotal = " + str(loss[i[0]]))
	i = np.argwhere(lossH == np.min(lossH)).flatten()
	print("Best AR model for high flows: AR("+str(i[0]+1)+"), with Jtotal = " + str(lossH[i[0]]))
	i = np.argwhere(lossL == np.min(lossL)).flatten()
	print("Best AR model for medium-low flows: AR("+str(i[0]+1)+"), with Jtotal = " + str(lossL[i[0]]))
	plt.show()





def plot_linearARXlosses(loss, lossH, lossL, Nref=None, Pref=None):
	rgba = getRGBA()
	FC = getFC()
	fig = plt.figure()
	spec = fig.add_gridspec(nrows=1, ncols=2)
	xN = np.arange(1, loss.shape[0]+1)
	xP = np.arange(1, loss.shape[1]+1)
	weights = np.array([1, 2, 1]) # loss, lossH, lossL

	# PLOT LOSSES FOR BEST X(P) and all AR(N)
	if Pref == None:
		idxBest = np.zeros(3)
		idxBest[0] = np.median( np.argsort(loss, axis=1)[:,0] )
		idxBest[1] = np.median( np.argsort(lossH, axis=1)[:,0] )
		idxBest[2] = np.median( np.argsort(lossL, axis=1)[:,0] )
		idxBest = int( np.average(idxBest, weights=weights) )
	else:
		idxBest = Pref - 1
	# plot losses
	ax0 = fig.add_subplot(spec[0, 0])
	ax0.plot(xN,loss[:,idxBest], '.-', c=rgba[16], label="Global loss $J_{FINAL}$")
	ax0.plot(xN,lossH[:,idxBest], '.-', c=rgba[1], label="$J_{FINAL}$ for high flows")
	ax0.plot(xN,lossL[:,idxBest], '.-', c=rgba[19], label="$J_{FINAL}$ for medium/low flows")
	# plot lowest points
	i = np.argsort(loss[:,idxBest])[0]
	ax0.plot(i+1,loss[i,idxBest], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossH[:,idxBest])[0]
	ax0.plot(i+1,lossH[i,idxBest], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossL[:,idxBest])[0]
	ax0.plot(i+1,lossL[i,idxBest], 'o', c=rgba[12], markersize=5, alpha=0.7)
	# Plot figure
	ax0.grid(linestyle='--', linewidth=0.5)
	ax0.set_title("Total losses ($J_{FINAL}$) for ARX(N," +str(idxBest+1)+ ")")
	ax0.set_ylabel("m3/s")
	ax0.set_xlabel("model order N (best P: " + str(idxBest+1) + ")")
	ax0.set_facecolor(getFC())
	ax0.legend()
	
	# PLOT LOSSES FOR BEST AR(N) and all X(P)
	if Nref == None:
		idxBest = np.zeros(3)
		idxBest[0] = np.median( np.argsort(loss, axis=0)[0,:] )
		idxBest[1] = np.median( np.argsort(lossH, axis=0)[0,:] )
		idxBest[2] = np.median( np.argsort(lossL, axis=0)[0,:] )
		idxBest = int( np.average(idxBest, weights=weights) )
	else:
		idxBest = Nref - 1
	# plot losses
	ax1 = fig.add_subplot(spec[0, 1])
	ax1.plot(xP,loss[idxBest,:], '.-', c=rgba[16])
	ax1.plot(xP,lossH[idxBest,:], '.-', c=rgba[1])
	ax1.plot(xP,lossL[idxBest,:], '.-', c=rgba[19])
	# plot lowest points
	i = np.argsort(loss[idxBest,:])[0]
	ax1.plot(i+1,loss[idxBest,i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossH[idxBest,:])[0]
	ax1.plot(i+1,lossH[idxBest,i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	i = np.argsort(lossL[idxBest,:])[0]
	ax1.plot(i+1,lossL[idxBest,i], 'o', c=rgba[12], markersize=5, alpha=0.7)
	# Plot figure
	ax1.grid(linestyle='--', linewidth=0.5)
	ax1.set_title("Total losses ($J_{FINAL}$) for ARX(" +str(idxBest+1)+ ",P)")
	ax1.set_ylabel("m3/s")
	ax1.set_xlabel("model order P (best N: " + str(idxBest+1) + ")")
	ax1.set_facecolor(getFC())

	# PRINT BEST INDICES OVERALL AND Jtotal VALUES
	i = np.argwhere(loss == np.min(loss)).flatten()
	if Nref != None: i[0] = Nref-1
	if Pref != None: i[1] = Pref-1
	print("Best ARX model globally: ARX("+str(i[0]+1)+","+str(i[1]+1)+"), with Jtotal = " + str(loss[i[0],i[1]]))
	i = np.argwhere(lossH == np.min(lossH)).flatten()
	if Nref != None: i[0] = Nref-1
	if Pref != None: i[1] = Pref-1
	print("Best ARX model for high flows: ARX("+str(i[0]+1)+","+str(i[1]+1)+"), with Jtotal = " + str(lossH[i[0],i[1]]))
	i = np.argwhere(lossL == np.min(lossL)).flatten()
	if Nref != None: i[0] = Nref-1
	if Pref != None: i[1] = Pref-1
	print("Best ARX model for medium-low flows: ARX("+str(i[0]+1)+","+str(i[1]+1)+"), with Jtotal = " + str(lossL[i[0],i[1]]))

	# SHOW PLOTS
	plt.show()




def plot_NNresults(valid_dataset, net, idx):
	rgba = getRGBA()
	FC = getFC()
	year = str(int(idx+1990))
	y_lab = []
	y_est = []
	net.to('cpu')
	for i in range(len(valid_dataset[:]["label"][:])):
		X = valid_dataset[i]["input"]
		Y_ = net(X) *valid_dataset[i]["mstd"] + valid_dataset[i]["ma"] 
		y_lab.append( valid_dataset[i]["label"].item() )
		y_est.append( Y_.item() )
	fig, [ax1, ax2, ax3] = plt.subplots(3)
	fig.suptitle("Validation year "+year)
	#
	ax1.plot(y_est, y_lab, '.', alpha=0.7, color=rgba[17])
	ax1.plot([0, max(y_est)], [0, max(y_est)], '--', linewidth=0.6, alpha=0.9, color=rgba[5])
	ax1.set_xlabel("Estimated flow [$m^3/s$]")
	ax1.set_ylabel("Measured flow [$m^3/s$]")
	ax1.grid(linestyle='--', linewidth=0.5)
	ax1.set_facecolor(getFC())
	#
	ax3.plot(y_lab, np.array(y_est) - np.array(y_lab), '.', alpha=0.8,  color=rgba[3])
	ax3.set_xlabel("Observed streamflow [$m^3/s$]")
	ax3.set_ylabel("Estimation - observation [$m^3/s$]")
	ax3.set_title("Prediction error")
	ax3.grid(linestyle='--', linewidth=0.5)
	ax3.set_facecolor(getFC())
	#
	ax2.plot(range(len(valid_dataset[:]["label"][:])), y_lab, '-', linewidth=1.2, alpha=0.99,  color=rgba[1])
	ax2.plot(range(len(valid_dataset[:]["label"][:])), y_est, '--', linewidth=1, alpha=0.85,  color=rgba[17])
	ax2.set_xlabel("Days")
	ax2.set_ylabel("Flow (blue predicted)")
	ax2.grid(linestyle='--', linewidth=0.5)
	ax2.set_facecolor(getFC())
	#
	plt.show()



''' ... '''