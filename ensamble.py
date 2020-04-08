from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# My Color class
class C:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    end = '\033[0m'
# Grabs data from csv
def getDataFromCsv(path):
    return pd.read_csv(path, ',').to_numpy()
# Returns a set of new classifiers
def getClassifiers():
    return DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3), GaussianNB(), RandomForestClassifier(), GradientBoostingClassifier()
# Detatches last element from arrays and returns as two arrays
def separateTargetsFromEnd(data):
    return np.array(list(map(lambda x: x[:-1], data))), np.array(list(map(lambda x: x[-1], data)))
# Detatches first element from arrays and returns as two arrays
def separateTargetsFromBeginning(data):
    return np.array(list(map(lambda x: x[:1], data))), np.array(list(map(lambda x: x[0], data)))
# Calculates the averate
def calcAvg(exp, act):
    ctr = 0
    for i in range(len(exp)):
        if exp[i] == act[i]:
            ctr = ctr + 1
    return ctr / len(exp)
# Display accuracies to the user
def display(msg, dt, kn, nb, rf, gb):
    print(msg)
    print(f"{C.purple}  DT Accuracy: {calcAvg(dt, yTest)}{C.end}")
    print(f"{C.blue}  KN Accuracy: {calcAvg(kn, yTest)}{C.end}")
    print(f"{C.green}  NB Accuracy: {calcAvg(nb, yTest)}{C.end}")
    print(f"{C.yellow}  RF Accuracy: {calcAvg(rf, yTest)}{C.end}")
    print(f"{C.red}  GB Accuracy: {calcAvg(gb, yTest)}{C.end}")
    print()
# Fit all classifiers to values
def fitAll(dt, kn, nb, rf, gb, xTrain, yTrain):
    dt.fit(xTrain, yTrain)
    kn.fit(xTrain, yTrain)
    nb.fit(xTrain, yTrain)
    rf.fit(xTrain, yTrain)
    gb.fit(xTrain, yTrain)
    return dt, kn, nb, rf, gb
# Predict values for all classifiers
def predAll(dt, kn, nb, rf, gb, xTest):
    dtPred = dt.predict(xTest)
    knPred = kn.predict(xTest)
    nbPred = nb.predict(xTest)
    rfPred = rf.predict(xTest)
    gbPred = gb.predict(xTest)
    return dtPred, knPred, nbPred, rfPred, gbPred
def classify(msg, dt, kn, nb, rf, gb, xTrain, yTrain, xTest, yTest):
    # Fit classifiers
    dt, kn, nb, rf, gb = fitAll(dt, kn, nb, rf, gb, xTrain, yTrain)
    # Use classifiers to predict 
    dtPred, knPred, nbPred, rfPred, gbPred = predAll(dt, kn, nb, rf, gb, xTest)
    # Display accuracies
    display(msg, dtPred, knPred, nbPred, rfPred, gbPred)

############## EXECUTION ################
########### Small dataset ###############
# Get classifiers
dt, kn, nb, rf, gb = getClassifiers()
# Separate target values from data
x, y = separateTargetsFromBeginning(getDataFromCsv('assets2/wcd.csv'))
# Split data into train and test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)
# Classify
classify("Small data set", dt, kn, nb, rf, gb, xTrain, yTrain, xTest, yTest)

########### Medium dataset ##############
# Separate target values from data
x, y = separateTargetsFromEnd(getDataFromCsv('assets2/osi.csv'))
# Split data into train and test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)
# Classify
classify("Medium data set", dt, kn, nb, rf, gb, xTrain, yTrain, xTest, yTest)

########### Large dataset ##############
# Separate target values from data
x, y = separateTargetsFromEnd(getDataFromCsv('assets2/avila1.csv'))
# Split data into train and test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)
# Classify
classify("Large data set", dt, kn, nb, rf, gb, xTrain, yTrain, xTest, yTest)