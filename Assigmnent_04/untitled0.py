# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:04:39 2020

@author: arinj
"""

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import DistanceMetric as DM
import scipy
import sympy 
import statsmodels.api as stats
import sklearn.naive_bayes as nb
from sklearn.naive_bayes import CategoricalNB

purchase = pd.read_csv("F:/Assigmnents/Machine Learning/Assigmnent_04/Purchase_Likelihood.csv",
                       delimiter=',', usecols = ['group_size', 'homeowner', 'married_couple', 'insurance'])

purchase = purchase.dropna()

feature = ['group_size', 'homeowner', 'married_couple']
target = ['insurance']

xTrain = purchase[feature].astype('category')
yTrain = purchase[target].astype('category')

model = CategoricalNB()
fitt = model.fit(xTrain, yTrain)

xTest = xTrain.groupby(feature).first().reset_index()
xTest= pd.DataFrame(xTest)
xTest


pro = model.predict_proba(xTest)

print(pro)