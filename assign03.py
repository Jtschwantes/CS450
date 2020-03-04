from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import functools as ft
import pandas as pd
import numpy as np
# My Color class
class C:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    end = '\033[0m'
def calcAvg(exp, act):
    ctr = 0
    for i in range(len(exp)):
        if exp[i] == act[i]:
            ctr += 1
    return ctr / len(exp)

##################### Utility functions ######################
def getDataFromCsv(path, sep):
    return pd.read_csv(path, sep).to_numpy()

def getUnique(data):
    newArr = []
    for row in data:
        newArr.append(list(np.unique(row)))
    return list(np.array(newArr))

def getLookup(unique):
    Dict, ctr = {}, 0
    for x in unique:
        Dict[x] = ctr
        ctr = ctr + 1
    return Dict

def getReverseLookup(unique):
    RDic, ctr = {}, 0
    for x in unique:
        RDic[ctr] = x
        ctr = ctr + 1
    return RDic

def encode(row, lookup):
    return list(map(lambda x: lookup[x], row))

def decode(row, rlookup):
    return list(map(lambda x: rlookup[x], row))

def squish(row, multiplier = 1):
    mini = np.min(row)
    maxi = np.max(row) - mini
    return list(map(lambda x: (x - mini) / maxi * multiplier, row))

def trueFalse(row):
    val = row[0]
    row1 = list(map(lambda x: True if x == val else False, row))
    return row1, [not i for i in row1]

def fixValues(col):
    mySum = sum(list(map(lambda x: float(x), 
        (filter(lambda x: x != '?', col)))))
    return [mySum if x == '?' else x for x in col]

def convertToFloat(col):
    return [float(i) for i in col]

def convertToInt(col):
    return [int(i) for i in col]

###################### Student CSVs ##########################
# Get Data
data = getDataFromCsv("assets/student-mat.csv", ";")
tran = data.transpose()
unique = getUnique(tran)

# Type of encoding -> 0 = String, 1 = Numeric
types = [0, 0, 1, 0, 0, # 1-5 
         0, 1, 1, 0, 0, # 6-10
         0, 0, 1, 1, 0, # 11-15
         0, 0, 0, 0, 0, # 16-20
         0, 0, 0, 1, 1, # 21-25
         1, 1, 1, 1]    # 26-29

# Encode all values
newData = []
for x in range(len(types)):
    if types[x] == 0:
        newData.append(squish(encode(tran[x], getLookup(unique[x]))))
    elif types[x] == 1:
        newData.append(squish(tran[x]))

# Format to train and target data
newData = np.array(newData).transpose()
rlookup = getReverseLookup(tran[-1])
targets = encode(tran[-1], getLookup(unique[-1]))

# Randomize the data
xTrain, xTest, yTrain, yTest = train_test_split(
    newData, targets, test_size = 0.3)

# Use regressor
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(xTrain, yTrain)
predictions = regressor.predict(xTest)
arr = []
for x in map(lambda i, j: abs(i - j), predictions, yTest):
    arr.append(x)

#################### Cars Csv #############################
# Get Data
data = getDataFromCsv("assets/car.csv", ",")
tran = data.transpose()
unique = getUnique(tran)

# Encode all values
newData = []
for x in range(len(data[0]) - 1):
    newData.append(squish(encode(tran[x], getLookup(unique[x]))))

# Format to train and target data
newData = np.array(newData).transpose()
rlookup = getReverseLookup(tran[-1])
targets = encode(tran[-1], getLookup(unique[-1]))

# Randomize the data
xTrain, xTest, yTrain, yTest = train_test_split(
    newData, targets, test_size = 0.3)

# Use classifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
accuracy = calcAvg(predictions, yTest)

#################### MGP Csv #############################
# Get Data
data = getDataFromCsv("assets/auto-mpg.csv", ',')
tran = data.transpose()
unique = getUnique(tran)

# Fix missing values
tran[3] = fixValues(tran[3])

# Types to convert into -> 0 = float, 1 = int
types = [0, 1, 0, 0, 0, 0, 1, 1]
for i in range(len(types)):
    if types[i] == 0:
        tran[i] = convertToFloat(tran[i])
    elif types[i] == 1:
        tran[i] = convertToInt(tran[i])

# Encode all values
newData = []
for x in range(len(data[0]) - 2):
    newData.append(squish(tran[x + 1]))

# Format to train and target data
newData = np.array(newData).transpose()
targets = tran[0]

# Randomize the data
xTrain, xTest, yTrain, yTest = train_test_split(
    newData, targets, test_size = 0.3)

# Use regressor
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(xTrain, yTrain)
predictions = regressor.predict(xTest)
arr2 = []
for x in map(lambda i, j: abs(i - j), predictions, yTest):
    arr2.append(x)

print(f"{C.blue}Car CSV accuracy:   {C.green}{accuracy}{C.end}")
print(f"{C.blue}Auto MGP CSV error: {C.green}{sum(arr2)/len(arr2)}{C.end}")
print(f"{C.blue}Student CSV error:  {C.green}{sum(arr)/len(arr)}{C.end}")