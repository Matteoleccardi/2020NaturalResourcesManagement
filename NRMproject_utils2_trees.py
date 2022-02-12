import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from NRMproject_utils0 import *
from NRMproject_plots import *
from NRMproject_utils1_ANN import *



def getTreesData(all_data, model_type="", model_order=[1], inlude_day_of_year=False, train=True, cross_validation_index=0, preprocessing=0, HL=False):
	dataset = Dataset01(all_data, 
						train=train,
                        model_type=model_type,
                        model_order=model_order,
                        inlude_day_of_year=inlude_day_of_year,
                        cross_validation_index=cross_validation_index,
                        preprocessing=preprocessing )
	
	if not HL:
		X = []
		Y = []
		for i in range(len(dataset)):
			X.append(dataset[i]["input"].tolist())
			Y.append(dataset[i]["label"].tolist())
		return X, Y
	else:
		X, XL, XH = [], [], []
		Y, YL, YH = [], [], []
		for i in range(len(dataset)):
			# All data
			X.append(dataset[i]["input"].tolist())
			Y.append(dataset[i]["label"].tolist())
			# Hogh flows
			if dataset[i]["label"] >= dataset[i]["ma"]+1.5*dataset[i]["mstd"]:
				XH.append(dataset[i]["input"].tolist())
				YH.append(dataset[i]["label"].tolist())
			# Low flows
			if dataset[i]["label"] <= dataset[i]["ma"]*0.5:
				XL.append(dataset[i]["input"].tolist())
				YL.append(dataset[i]["label"].tolist())
		return X, XL, XH, Y, YL, YH


def treeMSE(listYest, listYlabel):
	if len(listYest) != len(listYlabel): print("ERROR in treeMSE"); quit()
	Yest = np.array(listYest)
	Ylab = np.array(listYlabel)
	e = (Yest - Ylab)**2
	return np.mean(e)

def treeRMSE(listYest, listYlabel):
	return np.sqrt(treeMSE(listYest, listYlabel))


### BAGGED TREES

class baggedTree():
	def __init__(self, criterion,splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, numTrees=5):
		self.n = numTrees
		self.trees = []
		for i in range(self.n):
			self.trees.append(DecisionTreeRegressor(criterion=criterion,
							                           splitter=splitter,
							                           max_depth=max_depth,
						                               min_samples_split=min_samples_split, 
						                               min_samples_leaf=min_samples_leaf,
						                               max_leaf_nodes=max_leaf_nodes)
			)

	def fit(self, Xt, Yt):
		for i in range(self.n):
			# Get randomly selected data
			X, Y = [], []
			for j in range( int( len(Yt)*0.82 ) ):
				r = random.randrange(len(Yt))
				X.append(Xt[r])
				Y.append(Yt[r])
			# Fit tree on selected data
			self.trees[i].fit(X, Y)

	def predict(self, X):
		Y = 0
		for i in range(self.n):
			Y += self.trees[i].predict(X)
		Y /= self.n
		return Y

