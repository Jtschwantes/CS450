from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

file = open("assets/student-mat.csv", "r")
key = []
dic = {}

# Get number of columns
numCols = 0
for line in file:
    for value in line.split(";"):
        numCols = numCols + 1
    break

# Prepare new dictionary
for x in range(numCols):
    dic[x] = []

# Get unique list of values
ctr = 0
for line in file:
    for value in line.split(";"):
        datum = value.strip()
        if datum not in dic[ctr]:
            dic[ctr].append(datum)
        ctr = ctr + 1
        if ctr == numCols:
            ctr = 0
print(f"{C.yellow}{dic}")

# Prepare new dictionary of lookup values
lookup = {}
rlookup = {}
for x in range(numCols):
    lookup[x] = {}
    rlookup[x] = {}

# Prepare look up based off of values
counter = 0
counter2 = 0
for y in range(numCols):
    for val in dic[y]:
        lookup[counter2][val.strip()] = counter
        rlookup[counter2][counter] = val.strip()
        counter = counter + 1
    counter = 0
    counter2 = counter2 + 1
print(f"{C.blue}{lookup}")
print(f"{C.green}{rlookup}{C.end}")

file = pd.read_csv("assets/student-mat.csv")
data = file.to_numpy()

# Finally, prepare the data
newData = []
counter3 = 0
for line in data:
    newArr = []
    for val in line:
        val = val.strip()
        newArr.append(lookup[counter3][val])
        counter3 = counter3 + 1
    newData.append(newArr)
    counter3 = 0

# Get targets into their own array
targets = []
for line in newData:
    targets.append(line[-1])
for x in range(len(newData)):
    newData[x].pop()

xTrain, xTest, yTrain, yTest = train_test_split(
    newData, targets, test_size = 0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)
print(calcAvg(predictions, yTest))
# print(f"{C.red}{newData[10]}")
# print(f"{C.yellow}{newData[50]}")
# print(f"{C.green}{newData[100]}{C.end}")

