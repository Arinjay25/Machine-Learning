#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Queastion 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataFram = pd.read_csv("F:/Assigmnents/Machine Learning/Assigmnent_01/NormalSample.csv")


# In[40]:


# Queastion 1-a bin-width for the histogram of x
dataFram
x = dataFram['x']

qut1 = np.percentile(x,25)
qut3 = np.percentile(x,75)
#W = 2 (IQR) N-1/3
#IQR is the interquartile range (the 75th percentile minus the 25th percentile)
IQR = qut3 - qut1
N = len(x)
Bin_Width = 2*IQR*N**(-1/3)
print(Bin_Width)


# In[3]:


#Queastion 1-b What are the minimum and the maximum values of the field x?
print("Max value of X", np.max(x))
print("Min value of X", np.min(x))


# In[18]:


#Question 1-c Let a be the largest integer less than the minimum value of the field x, 
#and b be the smallest integer greater than the maximum value of the field x.  What are the values of a and b?
a = math.floor(np.min(x))
print("largest integer less than the minimum value of the field x: a=", a)
b = math.ceil(np.max(x))
print("smallest integer greater than the maximum value of the field x: b=", b)


# In[143]:


#Question 1-D 
def densityestimator(h):
    m = a + h/2
    allm = [] #mid points 
    allp = [] #density estimates
    wu = []
    for j in range(len(x)):
        allm.append(m)
        wu = []
        u = (x - m)/h
        for i in u:
            if i > -1/2 and i <= 1/2:
                wu.append(1)
            else:
                wu.append(0)
        if m > b:
            break
        else:
            m = m + h
        allp.append(sum(wu)/(N*h))
        wu.clear()
    print("List the coordinates of the density estimator for h =",h)        
    print(list(zip(allm,allp)))
    bins = int((b-a)/h)
    plt.hist(x,bins)
    plt.title('Histogram of Normal Sample h =' +str(h))
    plt.grid(axis="x")
    plt.show()
    print("Bin Width for h =",h,"Bins =",bins)
    print("\n")
densityestimator(0.25)

densityestimator(0.5)

densityestimator(1)

densityestimator(2)


# In[42]:


print("\n")
print("Among the four histograms, I can say h=0.5 is more close to Izenman bin-width (0.399) and it more seem likes Normal Distribution, Symmetric about x = 32")


# In[79]:


#Question 2 
#Question 2-a
def fiveNumberSummary(x):
    Min = np.min(x)
    Max = np.max(x)
    qut1 = np.percentile(x,25)
    qut3 = np.percentile(x,75)
    median = np.percentile(x,50)
    IQR = qut3-qut1
    print(" Minimum of X =",Min,"\n","First Quartile of X =",qut1,"\n","Median of X =",median,"\n","Third Quartile =",qut3,"\n","Maximum of X =",Max,"\n")
    print("IQR of X:",IQR)
    print("1.5 * IQR whiskers is :", 1.5*IQR)

    print("The lower whisker extends to the larger of Q1 – 1.5 * IQR =", qut1 - 1.5*IQR)
    print("The upper whisker extends to the smaller of Q3 + 1.5 * IQR =", qut3 + 1.5*IQR,"\n","\n")
    
fiveNumberSummary(x)


# In[80]:


#Question 2-b
x0 = []
x1 = []
for i in range(len(x)):
    if dataFram['group'][i] == 0:
        x0.append(x[i])
    else:
        x1.append(x[i])
print("Five Number Summary for X where group value is:'0'","\n")
fiveNumberSummary(x0)
print("Five Number Summary for X where group value is:'1'","\n")
fiveNumberSummary(x1)


# In[90]:


plt.boxplot(x, vert=0, patch_artist=True)
plt.title("Box plot of x")
plt.show()
print("\n")
print("From this box plot we can say it showing the same values as we calculated in 2.a), boxplot has correctly displayed the 1.5 IQR whiskers.")


# In[120]:


y = [x,x0,x1]
plt.boxplot(y, patch_artist=True)
plt.title("Box plot of Over all x  Group 0  Group 1")
plt.show()
def outliers(x):
    qut1 = np.percentile(x,25)
    qut3 = np.percentile(x,75)
    IQR = qut3-qut1
    outlier = []
    for k in x:
        if k < (qut1 - 1.5*IQR) or k > (qut3 + 1.5*IQR):
            outlier.append(k)
    print(outlier)


# In[121]:


print("outliers of x:")
outliers(x)
print("outliers of Group 0:")
outliers(x0)
print("outliers of Group 1:")
outliers(x1)


# In[125]:


#Question 3
df = pd.read_csv("F:/Assigmnents/Machine Learning/Assigmnent_01/Fraud.csv")


# In[129]:


#Question 3 a
fraud = df['FRAUD']
print(np.round((fraud.value_counts()[1]/len(fraud)*100),4))


# In[152]:


#Question 3 b
col_name = df.keys().tolist()
col_name.remove('CASE_ID')
col_name.remove('FRAUD')


# In[154]:


dffraud = df[df['FRAUD'] == 1]
dfnonfraud = df[df['FRAUD'] == 0]
def boxplots(name):
    fradulent = []
    nonfradulent = []
    for i in range(0,len(df[name])):
        if fraud[i] == 1:
            fradulent.append(df[name][i])
        else:
            nonfradulent.append(df[name][i])
    boxData = [nonfradulent,fradulent]
    plt.boxplot(boxData, vert=0, labels=[0,1], patch_artist= True)
    plt.title(name)
    plt.show()
for i in col_name:
    boxplots(i)


# In[161]:


#Question 3 c
from numpy import linalg as LA
x = np.matrix(df.drop(['CASE_ID','FRAUD'], axis=1))
xtx = x.transpose() * x
evalues, evects = LA.eigh(xtx)
print(evalues)
transf = evects * LA.inv(np.sqrt(np.diagflat(evalues)));
print("Transformation Matrix = \n", transf)

transf_x = x * transf;
print("The Transformed x = \n", transf_x)
xtx = transf_x.transpose() * transf_x;
print("Identity Matrix = \n", xtx)
xtx.shape

from scipy import linalg as LA2

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)


check = orthx.transpose().dot(orthx)
print("Identity Matrix = \n", check)


# In[162]:


#Question 3 d
from sklearn.neighbors import KNeighborsClassifier

trainData = df.drop(['CASE_ID','FRAUD'], axis=1)
target = df['FRAUD']

kNeigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbs = kNeigh.fit(trainData, target)
score_value = nbrs.score(x, target)
print(score_value)


# In[182]:


from sklearn.neighbors import NearestNeighbors as knn
inputData = [7500,15,3,127,2,2]
print("The Input values = ", inputData)
transfInputData = inputData * transf;
neighTarget = knn(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbsTarget = neighTarget.fit(transf_x)

neighborsTarget = nbsTarget.kneighbors(transfInputData, return_distance = False)
print("The five neighbors = ", neighborsTarget)


# In[185]:


print("Predicted Probability of Fraudulent ",nbs.predict(transf_x))

