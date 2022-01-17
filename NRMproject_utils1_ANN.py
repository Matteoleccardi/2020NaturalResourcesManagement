import numpy as np
import matplotlib
import matplotlib.pylab as plt

from NRMproject_utils0 import *
from NRMproject_plots import *

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn





# Create a custom dataset

class Dataset01(Dataset):
	def __init__(self, all_data, model_type="", model_order=[1], inlude_day_of_year=False, train=True, cross_validation_index=0, preprocessing=0):
		# Extract data - get data from the source
		''' every series is a numpy array 
			year  = all_data[:,0]
			month = all_data[:,1]
			day_m = all_data[:,2] # day number from month start (1-30)
			rain  = all_data[:,3] # mm/d
			flow  = all_data[:,4] # m3/d
			temp  = all_data[:,5] # °C
		'''
		self.all_data = all_data
		self.labels = all_data[:,4].copy()
		trainIndex = self.all_data[:,0] != (1990 + cross_validation_index)
		series = self.all_data[trainIndex,4].copy()
		self.flowMA = np.array([annualMovingAverage(series=series, semiwindow=6) for i in range(20)]).flatten()
		self.flowMSTD = np.array([np.sqrt(annualMovingVariance(series=series, semiwindow=6)) for i in range(20)]).flatten()
		# Other necessary stuff
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
		self.train=train
		''' cross_validation_index:
			the index indicating on which of the 20 years we are performing training
			(indexed 0 through 19)
		'''
		if (cross_validation_index < 0) or (cross_validation_index >19):
			print("Error: cross_validation_index out ob bounds."); quit()
		self.cv_idx = cross_validation_index
		self.doy = np.array([range(1,365+1,1) for i in range(20)]).reshape((365*20, 1))
		''' index representing the number of past time instants needed.
			It is also the index from which we start getting the observations (lables),
			in other words the global index at which (t+1) starts for the first time.
			It returns the index at which the labels (observations) data
			should start on. Es: AR(1) -> n_past=1; ARX(3,2,5) -> n_past=5
		'''
		self.n_past = int( np.max(self.model_order) )
		''' data preprocessing: integer indexing many different methods, declared as class methods.
			if 0 no preprocessing is perdormed on data
		'''
		self.preprocessing = preprocessing
		if self.preprocessing == 1:
			''' data preprocessing consists in  standardization and detrending, by using only the training dataset.
				The validation dataset is not considered as it would not be fair.
			'''
			''' Flow '''
			trainIndex = self.all_data[:,0] != (1990 + self.cv_idx)
			self.all_data[:,4] =  (self.all_data[:,4].copy() - self.flowMA) / self.flowMSTD
			''' Rain '''
			series = self.all_data[trainIndex,3].copy()
			ma = annualMovingAverage(series=series, semiwindow=6)
			ma = np.array([ma for i in range(20)]).flatten()
			mstd = np.sqrt(annualMovingVariance(series=series, semiwindow=6))
			mstd = np.array([mstd for i in range(20)]).flatten()
			self.all_data[:,3] =  (self.all_data[:,3].copy() - ma) / mstd
			''' Temp '''
			series = self.all_data[trainIndex,5].copy()
			ma = annualMovingAverage(series=series, semiwindow=6)
			ma = np.array([ma for i in range(20)]).flatten()
			mstd = np.sqrt(annualMovingVariance(series=series, semiwindow=6))
			mstd = np.array([mstd for i in range(20)]).flatten()
			self.all_data[:,5] =  (self.all_data[:,5].copy() - ma) / mstd
		''' inlude_day_of_year: if true, the last input will be the 1-365 day number
			of the forecasted day (t+1)
		'''
		self.inlude_day_of_year = inlude_day_of_year


	def __len__(self):
		cv_year = 1990 + self.cv_idx
		if self.train:
			select_idx = self.all_data[:,0] != cv_year
		else:
			select_idx = self.all_data[:,0] == cv_year
		return len( self.all_data[select_idx,0][self.n_past:] )

	def __getitem__(self, index):
		''' index = index to access one string of data (which will be input of the ANN).
					index starts from zero, which represent n_past in the complete time series array.
			input_data = data string (which will be the input tensor to the network)
			label[index] = unprocessed prediction
		'''
		if torch.is_tensor(index):
			index = index.tolist()
		# Trainin or validation indexes
		cv_year = 1990 + self.cv_idx
		if self.train:
			select_idx = self.all_data[:,0] != cv_year
		else:
			select_idx = self.all_data[:,0] == cv_year
		# Output data
		''' Order:
		           F(t) + F(t-1) +...+ F(t-N) + R(t+1) + R(t) +...+ T(t+1) + T(t) + ...
		    input_data array positions of model inputs (as represented in libe above):
		            0   +    1   +...+ ....
		'''
		''' Flow '''
		series = self.all_data[select_idx,4]
		s = self.n_past + index - self.model_order[0]
		e = self.n_past + index - 1
		input_data = np.flip(series[s:e+1])
		''' Rain '''
		series = self.all_data[select_idx,3]
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
		series = self.all_data[select_idx,5]
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
			input_data = np.append(input_data, self.doy[self.n_past + index] )
		''' Put input data into tensor '''
		input_data = torch.tensor(input_data.copy(), dtype=torch.float32)
		# Label (flow at time t+1)
		series = self.labels[select_idx]
		label = series[self.n_past + index]
		label = torch.tensor(label, dtype=torch.float32)
		stdlabel = ( series[self.n_past + index] - self.flowMA[self.n_past + index]) / self.flowMSTD[self.n_past + index]
		stdlabel = torch.tensor(stdlabel, dtype=torch.float32)
		# Day of the year of the label, estimated MA and MSTD from the training set
		doy_out  = torch.tensor(self.doy[self.n_past + index],  dtype=torch.float32)
		ma_out   = torch.tensor(self.flowMA[self.n_past + index],  dtype=torch.float32)
		mstd_out = torch.tensor(self.flowMSTD[self.n_past + index],  dtype=torch.float32)
		# Return structured data (tensor, float)
		return {"input": input_data, "label": label, "stdlabel": stdlabel, "doy": doy_out, "ma": ma_out, "mstd": mstd_out}

	def printDatasetInfo(self):
		print("Dataset created." )
		###



# Networks

class ANN(nn.Module):
	def __init__(self, input_size=1):
		super(ANN, self).__init__()
		if input_size < 1:
			quit()
		print("Created NN with input size ", input_size)
		# Input-dependent parameters
		size_o1 = int(input_size*2.2)
		# Layers
		self.shallow = nn.Sequential(
			nn.Linear(input_size, size_o1, bias=True),
			nn.LogSigmoid(),#nn.Tanh(), #nn.Sigmoid(),
			nn.Linear(size_o1, 1, bias=True)
		)
		self.deep = nn.Sequential(
			nn.Linear(input_size, size_o1, bias=True),
			nn.LogSigmoid(),
			nn.Linear(size_o1, int(1.5*input_size)+1, bias=True),
			nn.Tanh(),
			nn.Linear(int(1.5*input_size)+1, input_size+3, bias=True),
			nn.Sigmoid(),
			nn.Linear(input_size+3, 1, bias=True)
		)
		
	def forward(self, input_t):
		out = self.deep(input_t)
		return out.flatten()









# Training and validation ANN loops

def train_loop(dataloader, model, device, loss_fn, optimizer, verbose= True):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	train_loss = 0
	print("Training...")
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


def valid_loop(dataloader, model, device, loss_fn, verbose= True):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss = 0
	if verbose: print("Validating...")
	with torch.no_grad():
		for sample_batched in dataloader:
			X = sample_batched["input"].to(device)
			Y = sample_batched["label"].to(device)
			Y_ = model(X) *sample_batched["mstd"].to(device) + sample_batched["ma"].to(device)
			test_loss += loss_fn(Y_, Y).item()
	test_loss /= num_batches
	if verbose: print(f"Avg loss over batches: {test_loss:>8f}")
	return test_loss






# Helper functions
def get_modelOrdersToTest(Fmax_, Rmax_, i_rain, Tmax_, i_temp):
	orders_to_test = []
	for f in range(1,Fmax_+1):
		if (i_rain or Rmax_>0): # rain
			if Rmax_>0: # proer rain
				for r in range(1, Rmax_+1):
					if (i_temp or Tmax_>0): # also temp
						if Tmax_>0: # proper rain and also proer temp
							for t in range(1, Tmax_+1):
								orders_to_test.append([f, r, t])
						else: # proper rain and only improper temp
							orders_to_test.append([f, r, 0])
					else: # just proper rain
						orders_to_test.append([f, r])
			else: # only improper rain
				if (i_temp or Tmax_>0): # also temp
					if Tmax_>0: # only improper rain and also/only proer temp
						for t in range(1, Tmax_+1):
							orders_to_test.append([f, 0, t])
					else: # only improper rain and only improper temp
						orders_to_test.append([f, 0, 0])
				else: # just improper rain
					orders_to_test.append([f, 0])
		elif (not i_rain) and (Rmax_==0) and (i_temp or Tmax_>0): # just temp
			if Tmax_>0: # also/only proer
				for t in range(1, Tmax_+1):
					orders_to_test.append([f, t])
			else: # only improper
				orders_to_test.append([f, 0])
		else: # no rain, no temp 
			orders_to_test.append([f])
	return orders_to_test