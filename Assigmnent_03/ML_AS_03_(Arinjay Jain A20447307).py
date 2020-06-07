#!/usr/bin/env python
# coding: utf-8

# <br>
#     Arinjay Jain (A20447307)
#         </br>

# <br>Machine Learning Assignment-3 </br>

# Question 1

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

claimHistory = pandas.read_csv('F:\Assigmnents\Machine Learning\Assigmnent_03\claim_history.csv',
                            delimiter=',')


# In[2]:


print('Number of Missing Values:')
claimHistory.isna().sum()
## We do not have any missing values in our predictor('EDUCATION','OCCUPATION','CAR_TYPE') and target variable(CAR_USE)


# In[43]:


#Question 1 (a)
#test_size = 0.25 25% will assign into test partition 
claimHistory_train, claimHistory_test = train_test_split(claimHistory, test_size = 0.25, random_state = 60616)


# In[44]:


#total observation
claimHistory.shape[0]


# In[45]:


# number of observation in training 
trainCount = claimHistory_train.shape[0]
trainCount


# In[46]:


# number of observation in test 
testCount = claimHistory_test.shape[0]
testCount


# In[47]:


#Count 
claimHistory_train.groupby('CAR_USE').size()


# In[48]:


# proportions
claimHistory_train.groupby('CAR_USE').size()/trainCount


# In[9]:


#Question 1 (b)counts and proportions of the target variable in the Test partition
#Counts
claimHistory_test.groupby('CAR_USE').size()


# In[10]:


#proportions
claimHistory_test.groupby('CAR_USE').size()/testCount


# In[11]:


#Question 1 (c)
# What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?
probTrain = 0.75
probTest = 0.25
probTrainCommCar = 0.369014
probTestCommCar = 0.36413


# In[50]:


# By Bayes Theorem
a = probTrainCommCar * probTrain
b = probTestCommCar * probTest
prob_of_Comm_Car_in_Training = a/(a+b)
prob_of_Comm_Car_in_Training


# In[51]:


#Question 1 (d)
#What is the probability that an observation is in the Test partition given that CAR_USE = Private?
probTrainPrivateCar = 0.630986
probTestPrivateCar = 0.63587
c = probTrainPrivateCar * probTrain
d = probTestPrivateCar * probTest
prob_of_Private_Car_in_Test = d/(c+d)
prob_of_Private_Car_in_Test


# Question 2

# In[52]:


# Question 2
import itertools
def EntropyCalculator (data, split):    
        #data predictor in column 0 and target in column 1
        #split split set : combinations of predictors levels  
    countTable = pandas.crosstab(index = (data.iloc[:,0]).isin(split),columns = data.iloc[:,1],margins = False, dropna = True)
    fractionTable = countTable.div(countTable.sum(1), axis = 'index')
    totalRows = fractionTable.shape[0]
    totalColumns = fractionTable.shape[1]
    tableEntropy = 0
    tableN = 0
    for iRow in range(totalRows):
        rowEntropy = 0
        rowN = 0
        for iColumn in range(totalColumns):
            rowN += countTable.iloc[iRow, iColumn]
            proportion = fractionTable.iloc[iRow, iColumn]
            if (proportion > 0):
                rowEntropy -= (proportion * math.log2(proportion))
        tableEntropy += (rowN * rowEntropy)
        tableN += rowN
    tableEntropy = tableEntropy /  tableN
    return(tableEntropy)

def forNominalSplit (data):   
# data: predictor in column 0 and target in column 1
    catPred = set(data.iloc[:,0])
    nCatPred = len(catPred)
    treeResult = pandas.DataFrame(columns = ['Count of Left Child', 'Left Child', 'Right Child', 'Entropy'])
    for i in range(1, round((nCatPred+1)/2)):
        allComb_i = itertools.combinations(catPred, i)
        for comb in list(allComb_i):
            combComp = catPred.difference(comb)
            EV = EntropyCalculator(data, comb)
            treeResult = treeResult.append(pandas.DataFrame([[i, sorted(comb), sorted(combComp), EV]], 
                                           columns = ['Count of Left Child', 'Left Child', 'Right Child', 'Entropy']),
                                           ignore_index = True)

    treeResult = treeResult.sort_values(by = 'Entropy', axis = 0, ascending = True)
    return(treeResult)

def forOrdinalSplit (data, predValue):    
    catPred = set(predValue)
    nCatPred = len(catPred)
    treeResult = pandas.DataFrame(columns = ['Count of Left Child', 'Left Child', 'Right Child', 'Entropy'])
    for i in range(1, nCatPred):
        comb = list(predValue[0:i])
        combComp = list(predValue[i:nCatPred])
        EV = EntropyCalculator(data, comb)
        treeResult = treeResult.append(pandas.DataFrame([[i, comb, combComp, EV]], 
                                       columns = ['Count of Left Child', 'Left Child', 'Right Child', 'Entropy']),
                                       ignore_index = True)

    treeResult = treeResult.sort_values(by = 'Entropy', axis = 0, ascending = True)
    return(treeResult)


# In[53]:


#Question 2 
#Question 2 a)
#Root node entropy, from traning datasets
import math
rootEntropy = -(probTrainCommCar * math.log2(probTrainCommCar) + probTrainPrivateCar * math.log2(probTrainPrivateCar))
print("Entropy value of the root node", rootEntropy)


# In[54]:


# Question 2(b)
# for predictor EDUCATION
data = claimHistory_train[['EDUCATION', 'CAR_USE']]
levels=np.unique(claimHistory_train["EDUCATION"])
print(levels)
#we want in this order ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors']
treeEDUCATION = forOrdinalSplit(data,[levels[1], levels[3], levels[0], levels[4], levels[2]])
treeEDUCATION.iloc[0] #min entropy of education table


# In[55]:


# for predictor OCCUPATION
data = claimHistory_train[['OCCUPATION', 'CAR_USE']]
treeOCCUPATION = forNominalSplit(data)
treeOCCUPATION.iloc[0]


# In[56]:


# for predictor Car Type 
data = claimHistory_train[['CAR_TYPE', 'CAR_USE']]
treeCARTYPE = forNominalSplit(data)
treeCARTYPE.iloc[0]


# In[57]:


# Question 2(c)
data = claimHistory_train[['OCCUPATION', 'CAR_USE']]
split = list(['Blue Collar', 'Student', 'Unknown'])
EntropyCalculator(data, split)


# In[34]:


# Question 2(d)
# only those observations are in the left branch which has 'Blue Collar', 'Student', 'Unknown' OCCUPATION. 
left_Branch = claimHistory_train[claimHistory_train['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
print("Left Side if layer 1 \n")
data = left_Branch[['CAR_TYPE', 'CAR_USE']]
left_tree_CAR_TYPE = forNominalSplit(data)
print(left_tree_CAR_TYPE.iloc[0])

data = left_Branch[['OCCUPATION', 'CAR_USE']]
left_tree_OCCUPATION = forNominalSplit(data)
print(left_tree_OCCUPATION.iloc[0])

data = left_Branch[['EDUCATION', 'CAR_USE']]
left_tree_EDUCATION = forOrdinalSplit(data, [levels[1], levels[3], levels[0], levels[4], levels[2]])
print(left_tree_EDUCATION.iloc[0])


# only those observations are in the right branch which are not in left branch means other the 'Blue Collar', 'Student', 'Unknown' OCCUPATION. 
right_Branch = claimHistory_train[~claimHistory_train['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
print("\nRight Side if layer 1 \n")
data = right_Branch[['CAR_TYPE', 'CAR_USE']]
right_tree_CAR_TYPE = forNominalSplit(data)
print(right_tree_CAR_TYPE.iloc[0])

data = right_Branch[['OCCUPATION', 'CAR_USE']]
right_tree_OCCUPATION = forNominalSplit(data)
print(right_tree_OCCUPATION.iloc[0])

data = right_Branch[['EDUCATION', 'CAR_USE']]
right_tree_EDUCATION = forOrdinalSplit(data, [levels[1], levels[3], levels[0], levels[4], levels[2]])
print(right_tree_EDUCATION.iloc[0])


# In[58]:


# Question 2(e)
def leaf (row):
    if (numpy.isin(row['OCCUPATION'], ['Blue Collar', 'Student', 'Unknown'])):
        if (numpy.isin(row['EDUCATION'], ['Below High School'])):
            Leaf = 0
        else:
            Leaf = 1
    else:
        if (numpy.isin(row['CAR_TYPE'], ['Minivan', 'SUV', 'Sports Car'])):
            Leaf = 2
        else:
            Leaf = 3

    return(Leaf)

claimHistory_train = claimHistory_train.assign(Leaf = claimHistory_train.apply(leaf, axis = 1))

countTable = pandas.crosstab(index = claimHistory_train['Leaf'], columns = claimHistory_train['CAR_USE'],
                             margins = False, dropna = True)
probabilityCARUSE = countTable.div(countTable.sum(1), axis = 'index')

print('Counts:')
print(countTable)
print('\n')
print('probabilityCAR_USE:')
print(probabilityCARUSE)


# In[78]:


#Question 2 f)
def leaf_prob_Commercial (row):
    predProb = probabilityCARUSE.iloc[row['Leaf']]
    pCAR_USE_Commercial = predProb['Commercial']
    return(pCAR_USE_Commercial)

def leaf_prob_Private (row):
    predProb = probabilityCARUSE.iloc[row['Leaf']]
    pCAR_USE_Private = predProb['Private']
    return(pCAR_USE_Private)

claimHistory_train = claimHistory_train.assign(probTrainCAR_USE_Commercial = claimHistory_train.apply(leaf_prob_Commercial, axis = 1))

threshold = claimHistory_train.groupby('CAR_USE').size()/trainCount
threshold = threshold[0] #proportion of target Event(Commercial) value in the training partition 
print("Threshold: ", threshold)

fpr_train, tpr_train, thresholds_train =  metrics.roc_curve(claimHistory_train['CAR_USE'], claimHistory_train['probTrainCAR_USE_Commercial'], pos_label = 'Commercial')


# In[79]:


print(thresholds_train)
print(tpr_train-fpr_train)


# In[85]:


claimHistory_train


# <br>
#     Question 3
#     </br>

# In[80]:


# Question 3
claimHistory_test = claimHistory_test.assign(Leaf = claimHistory_test.apply(leaf, axis = 1))
claimHistory_test = claimHistory_test.assign(probCAR_USE_Commercial = claimHistory_test.apply(leaf_prob_Commercial, axis = 1))
claimHistory_test = claimHistory_test.assign(probCAR_USE_Private = claimHistory_test.apply(leaf_prob_Private, axis = 1))


# In[81]:


import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as metrics

threshold = claimHistory_train.groupby('CAR_USE').size()/trainCount
threshold = threshold[0] #proportion of target Event(Commercial) value in the training partition 
print("Threshold: ", threshold)
claimHistory_test['predictedCAR_USE'] = numpy.where(claimHistory_test['probCAR_USE_Commercial'] >= threshold, 'Commercial', 'Private')


# In[82]:


claimHistory_test


# In[91]:


# Question 3 a)
Matrix = metrics.confusion_matrix(claimHistory_test['CAR_USE'], claimHistory_test['predictedCAR_USE'])
print('Confusion Matrix')
print(Matrix)
print('\n')


# In[92]:


MCRate = 1.0 - metrics.accuracy_score(claimHistory_test['CAR_USE'], claimHistory_test['predictedCAR_USE'])
print('Misclassification Rate:', MCRate) 


# In[93]:


# Question 3 b)
fpr, tpr, thresholds = metrics.roc_curve(claimHistory_test['CAR_USE'], claimHistory_test['probCAR_USE_Commercial'], pos_label = 'Commercial')


# In[94]:


# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o',
         color = 'blue', linestyle = 'solid', label = 'True Positive')
plt.plot(cutoff, fpr, marker = 'o',
         color = 'red', linestyle = 'solid', label = 'False Positive')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)
plt.show()


# In[95]:


print(thresholds)
print(tpr-fpr)


# In[96]:


accuracy = metrics.accuracy_score(claimHistory_test['CAR_USE'], claimHistory_test['predictedCAR_USE'])
print('                  Accuracy: {:.13f}' .format(accuracy))
#Misclassification Rate
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))


# In[97]:


# Question 3 c)
# Calculate the Root Average Squared Error
nY = claimHistory_test['CAR_USE'].shape[0]
Y = numpy.array(claimHistory_test['CAR_USE'])
predProbY = numpy.array(claimHistory_test['probCAR_USE_Commercial'])
RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Commercial'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = numpy.sqrt(RASE/nY)
print('Root Average Squared Error')
print(RASE)


# In[98]:


# Question 3 d)
Y_true = 1.0 * numpy.isin(Y, ['Commercial'])
AUC = metrics.roc_auc_score(Y_true, predProbY)
print('          Area Under Curve: {:.13f}' .format(AUC))


# In[99]:


# Question 3 e)
#Gini Coefficient in the Test partition
Gini = 2*AUC -1 
print('          Gini Coefficient: {:.13f}' .format(Gini))


# In[100]:


# Question 3 f)

carUSE = numpy.array(claimHistory_test['CAR_USE'])

df=pandas.DataFrame({'x':carUSE, 'y':predProbY})
noneventProb = []
eventProb = []
for i in range(df.shape[0]):
    if df['x'][i] == 'Private':
        noneventProb.append(df['y'][i])
    else:
        eventProb.append(df['y'][i])


# In[101]:


Concordant = 0
Discordant = 0
Tied = 0
for i in range(len(eventProb)):
    for j in range (len(noneventProb)):
        if eventProb[i] > noneventProb[j]:
            Concordant = Concordant +1
        elif eventProb[i] == noneventProb[j]:
            Tied = Tied +1 
        elif eventProb[i] < noneventProb[j]:
            Discordant = Discordant +1
            
pair = Concordant + Discordant +Tied
print("Number of Pairs: ", pair)
print("Number of Concordant (C) pairs: ", Concordant)
print("Number of Discordant (D) pairs: ", Discordant)
print("Number of Tied (T) pairs: ", Tied)

print("Gini: ", (Concordant - Discordant)/pair)
print("GoodMan Gamma : ", ((Concordant - Discordant)/(Concordant + Discordant)))

#AUC using formula
Area = 0.5 + 0.5*((Concordant - Discordant)/pair)
print('Area Under Curve: {:.13f}' .format(AUC))


# In[102]:


# Question 3(g)
# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(claimHistory_test['CAR_USE'], claimHistory_test['probCAR_USE_Commercial'], pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()


# In[ ]:




