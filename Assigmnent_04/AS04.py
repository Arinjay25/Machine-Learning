# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 06:36:03 2020

@author: arinj
"""

import import_ipynb

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

dataFram = pd.read_csv("F:/Assigmnents/Machine Learning/Assigmnent_04/Purchase_Likelihood.csv")
# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

#A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


purchase = pd.read_csv("F:/Assigmnents/Machine Learning/Assigmnent_04/Purchase_Likelihood.csv",
                       delimiter=',', usecols = ['group_size', 'homeowner', 'married_couple', 'insurance'])

purchase = purchase.dropna()

# Specify Origin as a categorical variable
y = purchase['insurance'].astype('category')

# Specify group_size, homeowner and married_couple as categorical variables
xGS = pd.get_dummies(purchase[['group_size']].astype('category'))
xHO = pd.get_dummies(purchase[['homeowner']].astype('category'))
xMC = pd.get_dummies(purchase[['married_couple']].astype('category'))

# Intercept only model
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')

# Intercept + group_size
designX = stats.add_constant(xGS, prepend=True)
LLK_1GS, DF_1GS, fullParams_1GS = build_mnlogit (designX, y, debug = 'Y')
testDev_GS = 2 * (LLK_1GS - LLK0)
testDF_GS = DF_1GS - DF0
testPValue_GS = scipy.stats.chi2.sf(testDev_GS, testDF_GS)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev_GS)
print('  Degreee of Freedom = ', testDF_GS)
print('        Significance = ', testPValue_GS)


# Intercept + group_size + homeowner
designX = xGS
designX = designX.join(xHO)
designX = stats.add_constant(designX, prepend=True)
LLK_1GS_1HO, DF_1GS_1HO, fullParams_1GS_1HO = build_mnlogit (designX, y, debug = 'Y')
testDev_GS_HO = 2 * (LLK_1GS_1HO - LLK_1GS)
testDF_GS_HO = DF_1GS_1HO - DF_1GS
testPValue_GS_HO = scipy.stats.chi2.sf(testDev_GS_HO, testDF_GS_HO)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev_GS_HO)
print('  Degreee of Freedom = ', testDF_GS_HO)
print('        Significance = ', testPValue_GS_HO)

# Intercept + group_size + homeowner + married_couple
design_x = xGS
design_x = design_x.join(xHO)
design_x = design_x.join(xMC)
design_x = stats.add_constant(design_x, prepend=True)
LLK3_GS_HO_MC, DF3_GS_HO_MC, full_params3_GS_HO_MC = build_mnlogit (design_x, y, debug = 'Y')
testDev_GS_HO_MC = 2 * (LLK3_GS_HO_MC - LLK_1GS_1HO)
testDF_GS_HO_MC = DF3_GS_HO_MC - DF_1GS_1HO
testPValue_GS_HO_MC = scipy.stats.chi2.sf(testDev_GS_HO_MC, testDF_GS_HO_MC)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_GS_HO_MC)
print('  Degrees of Freedom = ', testDF_GS_HO_MC)
print('        Significance = ', testPValue_GS_HO_MC)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner
design_x = xGS
design_x = design_x.join(xHO)
design_x = design_x.join(xMC)

# Create the columns for the group_size * homeowner interaction effect
x_GSHO = create_interaction(xGS, xHO)
design_x = design_x.join(x_GSHO)
design_x = stats.add_constant(design_x, prepend=True)
LLK4_GS_HO_MC_GSHO, DF4_GS_HO_MC_GSHO, full_params4_GS_HO_MC_GSHO = build_mnlogit(design_x, y, debug='Y')
testDev_GS_HO_MC_GSHO = 2 * (LLK4_GS_HO_MC_GSHO - LLK3_GS_HO_MC)
testDF_GS_HO_MC_GSHO = DF4_GS_HO_MC_GSHO - DF3_GS_HO_MC
testPValue_GS_HO_MC_GSHO = scipy.stats.chi2.sf(testDev_GS_HO_MC_GSHO, testDF_GS_HO_MC_GSHO)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_GS_HO_MC_GSHO)
print('  Degrees of Freedom = ', testDF_GS_HO_MC_GSHO)
print('        Significance = ', testPValue_GS_HO_MC_GSHO)


# group_size * married_couple
# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple
design_x = xGS
design_x = design_x.join(xHO)
design_x = design_x.join(xMC)

# Create the columns for the group_size * homeowner interaction effect
x_GSHO = create_interaction(xGS, xHO)
design_x = design_x.join(x_GSHO)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_GSMC = create_interaction(xGS, xMC)
design_x = design_x.join(x_GSMC)
design_x = stats.add_constant(design_x, prepend=True)
LLK5_GS_HO_MC_GSHO_GSMC, DF5_GS_HO_MC_GSHO_GSMC, full_params5_GS_HO_MC_GSHO_GSMC = build_mnlogit(design_x, y, debug='Y')
testDev_GS_HO_MC_GSHO_GSMC = 2 * (LLK5_GS_HO_MC_GSHO_GSMC - LLK4_GS_HO_MC_GSHO)
testDF_GS_HO_MC_GSHO_GSMC = DF5_GS_HO_MC_GSHO_GSMC - DF4_GS_HO_MC_GSHO
testPValue_GS_HO_MC_GSHO_GSMC = scipy.stats.chi2.sf(testDev_GS_HO_MC_GSHO_GSMC, testDF_GS_HO_MC_GSHO_GSMC)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_GS_HO_MC_GSHO_GSMC)
print('  Degrees of Freedom = ', testDF_GS_HO_MC_GSHO_GSMC)
print('        Significance = ', testPValue_GS_HO_MC_GSHO_GSMC)



#  Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
design_x = xGS
design_x = design_x.join(xHO)
design_x = design_x.join(xMC)

# Create the columns for the group_size * homeowner interaction effect
x_GSHO = create_interaction(xGS, xHO)
design_x = design_x.join(x_GSHO)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_GSMC = create_interaction(xGS, xMC)
design_x = design_x.join(x_GSMC)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the homeowner * married_couple interaction effect
x_HOMC = create_interaction(xHO, xMC)
design_x = design_x.join(x_HOMC)
design_x = stats.add_constant(design_x, prepend=True)

LLK6_GS_HO_MC_GSHO_GSMC_HOMC, DF6_GS_HO_MC_GSHO_GSMC_HOMC, full_params6_GS_HO_MC_GSHO_GSMC_HOMC = build_mnlogit(design_x, y, debug='N')
testDev_GS_HO_MC_GSHO_GSMC_HOMC = 2 * (LLK6_GS_HO_MC_GSHO_GSMC_HOMC - LLK5_GS_HO_MC_GSHO_GSMC)
testDF_GS_HO_MC_GSHO_GSMC_HOMC = DF6_GS_HO_MC_GSHO_GSMC_HOMC - DF5_GS_HO_MC_GSHO_GSMC
testPValue_GS_HO_MC_GSHO_GSMC_HOMC = scipy.stats.chi2.sf(testDev_GS_HO_MC_GSHO_GSMC_HOMC, testDF_GS_HO_MC_GSHO_GSMC_HOMC)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_GS_HO_MC_GSHO_GSMC_HOMC)
print('  Degrees of Freedom = ', testDF_GS_HO_MC_GSHO_GSMC_HOMC)
print('        Significance = ', testPValue_GS_HO_MC_GSHO_GSMC_HOMC)


#Question 1 a
full_params6_GS_HO_MC_GSHO_GSMC_HOMC


#Question 1 b
print("Degree of Freedom =",DF6_GS_HO_MC_GSHO_GSMC_HOMC)

print("Log-Likelihood = ", LLK6_GS_HO_MC_GSHO_GSMC_HOMC)


#Question 1 d
fiIndex1 = -(math.log10(testPValue_GS))
#fiIndex2 = -(math.log10(testPValue_GS_HO)) --- positive Infinity because log(0)= -infinity
fiIndex3 = -(math.log10(testPValue_GS_HO_MC))
fiIndex4 = -(math.log10(testPValue_GS_HO_MC_GSHO))
fiIndex5 = -(math.log10(testPValue_GS_HO_MC_GSHO_GSMC))
fiIndex6 = -(math.log10(testPValue_GS_HO_MC_GSHO_GSMC_HOMC))
print("Feature Importance Index for model:(Intercept + group_size) =", fiIndex1)
#print('Feature Importance Index for model:(Intercept + group_size + homeowner) =', fiIndex2)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple) =", fiIndex3)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner) =", fiIndex4)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple) =",fiIndex5)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple) =", fiIndex6)





#Question 2 a

#  Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
design_x = xGS
design_x = design_x.join(xHO)
design_x = design_x.join(xMC)

# Create the columns for the group_size * homeowner interaction effect
x_GSHO = create_interaction(xGS, xHO)
design_x = design_x.join(x_GSHO)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_GSMC = create_interaction(xGS, xMC)
design_x = design_x.join(x_GSMC)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the homeowner * married_couple interaction effect
x_HOMC = create_interaction(xHO, xMC)
design_x = design_x.join(x_HOMC)
design_x = stats.add_constant(design_x, prepend=True)

reduced_form, inds = sympy.Matrix(design_x.values).rref()

# Extract only the non-redundant columns for modeling
X = design_x.iloc[:, list(inds)]


print('XXXXXXX:::',X)

logit = stats.MNLogit(y, design_x)
this_fit = logit.fit(method='newton', full_output=True, maxiter=30, tol=1e-8)
#logit.score(this_fit.)


gs_d = [1,2,3,4]
ho_d = [0,1]
mc_d = [0,1]
x_combos = []

for i in gs_d:
    for j in ho_d:
        for k in mc_d:
            data = [i,j,k]
            x_combos = x_combos + [data]

x_df = pd.DataFrame(x_combos, columns=['group_size','homeowner','married_couple'])
x_gs = pd.get_dummies(x_df[['group_size']].astype('category'))
x_ho = pd.get_dummies(x_df[['homeowner']].astype('category'))
x_mc = pd.get_dummies(x_df[['married_couple']].astype('category'))
x_design = x_gs
x_design = x_design.join(x_ho)
x_design = x_design.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
x_design = x_design.join(x_gsho)
x_design = stats.add_constant(x_design, prepend=True)
# Create the columns for the group_size * married_couple interaction effect
x_gsmc = create_interaction(x_gs, x_mc)
x_design = x_design.join(x_gsmc)
x_design = stats.add_constant(x_design, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
x_design = x_design.join(x_homc)
x_design = stats.add_constant(x_design, prepend=True)
Insuranc_pred = this_fit.predict(exog = x_design)
Insuranc_pred = pd.DataFrame(Insuranc_pred, columns = ['Insurance(0)', 'Insurance(1)','Insurance(2)'])
print(Insuranc_pred)