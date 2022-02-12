import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from NRMproject_utils0 import *
from NRMproject_plots import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn







# Networks

class ANN(nn.Module):
	def __init__(self, input_size=1):
		super(ANN, self).__init__()
		if input_size < 1:
			quit()
		self.param_lr_gamma = {"input_size": input_size}
		
		# Linear NN
		self.linear = nn.Sequential(
			nn.Linear(input_size, 1, bias=True)
		)
		self.param_lr_gamma["linear"] = [100.0e-3, 0.96]
		
		# Linear NN with ReLU activation function
		s = max([input_size, 3])
		self.shallowLinear = nn.Sequential(
			nn.Linear(input_size, s+1, bias=True),
			nn.ReLU(),
			nn.Linear(s+1, 1, bias=True)
		)
		self.param_lr_gamma["shallowLinear"] = [90.0e-3, 0.96]

		# Nonlinear sigmoid NN with ReLU activation function 
		s = max([input_size, 3])
		self.shallowNonlinear = nn.Sequential(
			nn.Linear(input_size, s+1, bias=True),
			nn.Sigmoid(), # Sigmoid() and Tanh() are very similar
			nn.ReLU(), # ELU() has very similar performance
			nn.Linear(s+1, 1, bias=True)
		)
		self.param_lr_gamma["shallowNonlinear"] = [50.0e-3, 0.96]

		# Deep Linear NN (composed of only linear layers and activation function ELU
		s = max([input_size, 5])
		self.deepLinear = nn.Sequential(
			nn.Linear(input_size, s, bias=True),
			nn.ReLU(),
			nn.Linear(s, s, bias=True),
			nn.ReLU(),
			nn.Linear(s, s+2, bias=True),
			nn.ReLU(),
			nn.Linear(s+2, 3, bias=True),
		)
		self.param_lr_gamma["deepLinear"] = [70.0e-3, 0.94]

		# Deep Nonlinear NN
		s = int(input_size*1.2)+2
		self.deepNonlinear = nn.Sequential(
			nn.Linear(input_size, input_size, bias=True),
			nn.Tanh(),
			nn.ELU(),
			nn.Linear(input_size, s, bias=True),
			nn.Sigmoid(),
			nn.ELU(),
			nn.Linear(s, 3, bias=True)			
		)
		self.param_lr_gamma["deepNonlinear"] = [100.0e-3, 0.94]

		# Very deep Nonlinear NN
		s = int(input_size*1.2)+2
		self.veryDeepNonlinear = nn.Sequential(
			nn.Linear(input_size, input_size, bias=True),
			nn.Tanh(),
			nn.ReLU(),
			nn.Linear(input_size, s, bias=True),
			nn.Sigmoid(),
			nn.Linear(s, s+5, bias=True),
			nn.LeakyReLU(0.2),
			nn.Linear(s+5, input_size, bias=True),
			nn.Sigmoid(),
			nn.Linear(input_size, input_size, bias=True),
			nn.ELU(),
			nn.Linear(input_size, 1, bias=True)
		)
		self.param_lr_gamma["veryDeepNonlinear"] = [110.0e-3, 0.94]

		self.bil = nn.Bilinear(3, 3, 1, bias=True)


	def forward(self, input_t):
		out = self.deepNonlinear(input_t)
		
		out2 = self.deepLinear(input_t)
		out = self.bil(out, out2)

		return out.flatten()









# Training and validation ANN loops

def train_loop(dataloader, model, device, loss_fn, optimizer, verbose= True):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	train_loss = 0
	if verbose: print("Training...")
	for batch_idx, sample_batched in enumerate(dataloader):
		# Compute prediction and loss
		X = sample_batched["input"].to(device)
		Y = sample_batched["stdlabel"].to(device)
		Y_ = model(X)
		loss = loss_fn(Y_, Y)
		train_loss += loss.item()
		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# Print output
		if verbose: 
			print(f"{(batch_idx / num_batches)*100:.0f}% completed. ", end = "")
			if batch_idx != num_batches-1:
				print("", end = "\r")
	train_loss /= num_batches
	if verbose: print(f"Avg loss over batches: {train_loss:>8f}")
	return train_loss


def valid_loop(dataloader, model, device, loss_fn, verbose=True, high_flow_loss=False):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss = 0
	test_loss_H = 0
	if verbose: print("Validating...")
	with torch.no_grad():
		for sample_batched in dataloader:
			X = sample_batched["input"].to(device)
			Y = sample_batched["label"].to(device)
			st= sample_batched["mstd"].to(device)
			m = sample_batched["ma"].to(device)
			Y_ = model(X) * st + m
			test_loss += loss_fn(Y_, Y).item()
			if high_flow_loss:
				l = 0
				w = 2.1
				while l < 10:
					w -= 0.05
					Y_h = Y_[Y>=torch.mean(m)+w*torch.mean(st)]
					Yh  = Y[Y>=torch.mean(m)+w*torch.mean(st)]
					l = len(Y_h)
				test_loss_H += loss_fn(Y_h, Yh).item()
	test_loss /= num_batches
	test_loss_H /= num_batches
	if verbose: print(f"Avg loss over batches: {test_loss:>8f}")
	if high_flow_loss:
		return test_loss, test_loss_H
	else:
		return test_loss


















# Create a custom dataset

class Dataset01(Dataset):
	def __init__(self, all_data, model_type="", model_order=[1], inlude_day_of_year=False, train=True, cross_validation_index=0, preprocessing=0):
		# Get all data into the class
		''' (all_data must be a numpy array) 
			year  = all_data[:,0]
			month = all_data[:,1]
			day_m = all_data[:,2] # day number from month start (1-30)
			rain  = all_data[:,3] # mm/d
			flow  = all_data[:,4] # m3/d
			temp  = all_data[:,5] # °C
		'''
		self.all_data = all_data

		# Get unprocessed labels
		self.labels = all_data[:,4].copy()
		
		# Get model type
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
		if model_type == "": self.model_type = 0
		elif model_type == "F_": self.model_type = 0
		elif model_type == "F_R_PRO": self.model_type = 1
		elif model_type == "F_R_IMP": self.model_type = 2
		elif model_type == "F_T_PRO": self.model_type = 3
		elif model_type == "F_T_IMP": self.model_type = 4
		elif model_type == "F_R_PRO_T_PRO": self.model_type = 5
		elif model_type == "F_R_IMP_T_PRO": self.model_type = 6
		elif model_type == "F_R_IMP_T_IMP": self.model_type = 7
		else: print("Unknown model type - assumed flow"); self.model_type = 0
		self.mods_impRain = np.array([2, 6, 7])
		self.mods_proRain = np.array([1, 5])
		self.mods_impTemp = np.array([4, 7])
		self.mods_proTemp = np.array([3, 5, 6])
		
		# Get model order
		''' model_order: the order of the models. Monodimensional list of integers
			The sequence must be consistent with the order in the model_type strings
			For purely improper models, use 0. Proper models are not allowed indexes < 1
			es: 
				model_type = "F_R_IMP_T_PRO"
				model_order = [4, 2, 3]
							4 past data of flow: (t, t-1, ..., t-3)
							1 present data for rain (t+1) + 2 past data (t, t-1)
							3 past data for temperature: (t, t-1, t-2)
		'''
		self.model_order = np.array(model_order)
		''' important to format the output of data,
			which is different between training and validation
			True: data for training
			False: data for validation
		'''
		
		# Get training/validation flag
		self.train=train
		
		# Get cross-validation index
		''' cross_validation_index:
			the index indicating on which of the 20 years we are performing training
			(indexed 0 through 19) where 0=1990 and 19=2009
		'''
		if (cross_validation_index < 0) or (cross_validation_index >19):
			print("Error: cross_validation_index out ob bounds."); quit()
		self.cv_idx = cross_validation_index
		self.cv_year = 1990 + self.cv_idx

		# Get dataset selection indices (Train/Val)
		self.select_idx = self.all_data[:,0] != self.cv_year if self.train else self.all_data[:,0] == self.cv_year


		# Get day of the year array made of 20 years
		self.doy = np.array([range(1,365+1,1) for i in range(20)]).reshape((365*20, 1))
		
		# Get numer of past days deeded for the model
		''' index representing the number of past time instants needed.
			It is also the index from which we start getting the observations (lables),
			in other words the global index at which (t+1) starts for the first time.
			It returns the index at which the labels (observations) data
			should start on. Es: AR(1) -> n_past=1; ARX(3,2,5) -> n_past=5
		'''
		self.n_past = int( np.max(self.model_order) )
		
		# Get training index
		''' boolean array representing, for each day, if the day is needed for training (True) or not (False)'''
		self.trainIndex = self.all_data[:,0] != (1990 + self.cv_idx)

		# Get moving average and moving standard deviation for the 20 years
		''' comouted only from training data to be consistent with the training-validation paradigm '''
		series = self.all_data[self.trainIndex,4].copy()
		self.flowMA = np.array([annualMovingAverage(series=series, semiwindow=6) for i in range(20)]).flatten()
		self.flowMSTD = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()

		# Flag to include day of year at the tail of input
		''' inlude_day_of_year: if true, the last input will be the 1-365 day number
			of the forecasted day (t+1)
		'''
		self.inlude_day_of_year = inlude_day_of_year

		# Data preprocessing
		''' if 0 no preprocessing is perdormed on data. 
			if 1, standard detrending and standardization is performed
		'''
		self.preprocessing = preprocessing
		if self.preprocessing == 1:
			''' Flow '''
			self.all_data[:,4] =  (self.all_data[:,4].copy() - self.flowMA) / self.flowMSTD
			self.all_data[:,4] = self.all_data[:,4].copy()
			''' Rain '''
			series = self.all_data[self.trainIndex,3].copy()
			ma = np.array([annualMovingAverage(series=series, semiwindow=6) for i in range(20)]).flatten()
			mstd = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()
			self.all_data[:,3] =  (self.all_data[:,3].copy() - ma) / mstd
			''' Temp '''
			series = self.all_data[self.trainIndex,5].copy()
			ma = np.array([annualMovingAverage(series=series, semiwindow=6) for i in range(20)]).flatten()
			mstd = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()
			self.all_data[:,5] =  (self.all_data[:,5].copy() - ma) / mstd
		elif self.preprocessing == 2:
			''' Flow '''
			self.all_data[:,4] =  self.all_data[:,4].copy() / self.flowMSTD
			self.all_data[:,4] = self.all_data[:,4].copy()
			''' Rain '''
			series = self.all_data[self.trainIndex,3].copy()
			mstd = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()
			self.all_data[:,3] =  self.all_data[:,3].copy() / mstd
			''' Temp '''
			series = self.all_data[self.trainIndex,5].copy()
			mstd = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()
			self.all_data[:,5] =  self.all_data[:,5].copy() / mstd
			
		# Extract data - get data from the source
		''' create an indexable array to extract data from the dataset faster
				rows: index of input and labels to extract (in __getitem__)
				columns: ordered input data:
					output logical order:
				           F(t) + F(t-1) +...+ F(t-N) + R(t+1) + R(t) +...+ T(t+1) + T(t) + ... + doy
				    input_data array positions of model inputs (as represented in libe above):
				            0   +    1   +...+ ....
		'''
		self.outputlist = []
		for index in range(0, len(self)): 
			''' Flow '''
			series = self.all_data[self.select_idx,4]
			s = self.n_past + index - self.model_order[0]
			e = self.n_past + index - 1
			input_data = np.flip(series[s:e+1])
			''' Rain '''
			series = self.all_data[self.select_idx,3]
			''' improper '''
			if (self.model_type == self.mods_impRain).any():
				input_data = np.append(input_data, series[self.n_past + index])
			''' proper '''
			condition = (self.model_type==self.mods_impRain).any() and (self.model_order[1] != 0)
			condition = condition or (self.model_type==self.mods_proRain).any()
			if condition:
				s = self.n_past + index - self.model_order[1]
				e = self.n_past + index - 1
				input_data = np.append(input_data, np.flip(series[s:e+1]) )
			''' Temperature '''
			series = self.all_data[self.select_idx,5]
			''' improper '''
			if (self.model_type == self.mods_impTemp).any():
				input_data = np.append(input_data, series[self.n_past + index])
			''' proper '''
			condition = (self.model_type==self.mods_impTemp).any() and (self.model_order[-1] != 0)
			condition = condition or (self.model_type==self.mods_proTemp).any()
			if condition:
				s = self.n_past + index - self.model_order[-1]
				e = self.n_past + index - 1
				input_data = np.append(input_data, np.flip(series[s:e+1]) )
			''' Day of year: value that could be useful for predicting outcomes knowing the day of the year (1-365)
				The day is meant as the day we want to predict (t+1, not t)
			'''
			if self.inlude_day_of_year:
				alpha = 2*np.pi * self.doy[self.n_past + index] / 365
				d_sin = np.sin(alpha)
				d_cos = np.cos(alpha)
				input_data = np.append(input_data,  d_sin)
				input_data = np.append(input_data,  d_cos)
			''' get input data row into a list containing all indexable inputs '''
			
			self.outputlist.append(
				torch.tensor(input_data.copy(), dtype=torch.float32)
				)
		
		''' Do the same for unprocessed labels and processed labels
			this allows slicing and also a much much faster training
		'''
		self.labels = self.labels[self.select_idx].copy()
		end = 365*19 if self.train else 365
		self.stdlabels = ( self.labels[:].copy() \
				- self.flowMA[:end].copy()) \
				/ self.flowMSTD[:end].copy()
		self.stdlabels = torch.tensor(self.stdlabels[self.n_past:].copy(), dtype=torch.float32)
		self.labels = torch.tensor(self.labels[self.n_past:].copy(), dtype=torch.float32)
		''' Same for other outputs '''
		self.doy  = torch.tensor(self.doy[self.n_past:].copy(), dtype=torch.float32)
		self.flowMA = torch.tensor(self.flowMA[self.n_past:].copy(), dtype=torch.float32)
		self.flowMSTD = torch.tensor(self.flowMSTD[self.n_past:].copy(), dtype=torch.float32)
		
		# Clear up some memory from stuff that are no longer needed
		''' to do '''
		

	def __len__(self):
		return len( self.all_data[self.select_idx,0][self.n_past:] )


	def __getitem__(self, index):
		''' index = index to access one string of data (which will be input of the ANN).
					index starts from zero, which represent n_past in the complete time series array.
			input_data = data string (which will be the input tensor to the network)
			label[index] = unprocessed prediction
		'''
		if torch.is_tensor(index):
			index = index.tolist()
		# Return structured data (tensor, float)
		return {"input": self.outputlist[index], "label": self.labels[index], "stdlabel": self.stdlabels[index], "doy": self.doy[index], "ma": self.flowMA[index], "mstd": self.flowMSTD[index]}

	def printDatasetInfo(self):
		print("Dataset created." )
		###



















# Helper functions

def get_modelOrdersToTest(Fmax_, Rmax_, i_rain, Tmax_, i_temp):
	orders_to_test = []
	for f in range(1,Fmax_+1):
		if (i_rain or Rmax_>0): # rain
			if Rmax_>0: # proer rain
				s = 0 if i_rain else 1
				for r in range(s, Rmax_+1):
					if (i_temp or Tmax_>0): # also temp
						if Tmax_>0: # proper rain and also proer temp
							s = 0 if i_temp else 1
							for t in range(s, Tmax_+1):
								orders_to_test.append([f, r, t])
						else: # proper rain and only improper temp
							orders_to_test.append([f, r, 0])
					else: # just proper rain
						orders_to_test.append([f, r])
			else: # only improper rain
				if (i_temp or Tmax_>0): # also temp
					if Tmax_>0: # only improper rain and also/only proer temp
						s = 0 if i_temp else 1
						for t in range(s, Tmax_+1):
							orders_to_test.append([f, 0, t])
					else: # only improper rain and only improper temp
						orders_to_test.append([f, 0, 0])
				else: # just improper rain
					orders_to_test.append([f, 0])
		elif (not i_rain) and (Rmax_==0) and (i_temp or Tmax_>0): # just temp
			if Tmax_>0: # also/only proer
				s = 0 if i_temp else 1
				for t in range(s, Tmax_+1):
					orders_to_test.append([f, t])
			else: # only improper
				orders_to_test.append([f, 0])
		else: # no rain, no temp 
			orders_to_test.append([f])
	return orders_to_test






















