from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import statistics as stats
import numpy as np

# # Show the x, y, and names respectively
# print(iris.data)
# print(iris.target)
# print(iris.target_names)
iris = datasets.load_iris()

# My Color class
class C:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    end = '\033[0m'

# My decision tree class
class tree:
    def __init__(self):
        self.xTrain = []
        self.yTrain = []
    # Fit doesn't really do much
    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain
    # Finds the K nearest neighbors
    def predict(self, xTest, k):
        return 1

# Function to get accuracy
# Formula for accuracy: TP + TN / Everything
# def calcAvg(exp, act):
#     ctr = 0
#     for i in range(len(exp)):
#         if exp[i] == act[i]:
#             ctr = ctr + 1
#     return ctr / len(exp)

# Reset data
xTrain, xTest, yTrain, yTest = train_test_split(
    iris.data, iris.target, test_size = 0.3)

# My implementation
tree = tree()
tree.fit(xTrain, yTrain)
results = tree.predict(xTest, 5)

# my_accuracy = calcAvg(results, yTest)


# Prints the final decision
# print()
# if score > 0:
#     print(f"{C.green}---My classifier was correct for more values of k---{C.end}")
# elif score == 0:
#     print(f"{C.yellow}---Our classifiers tied---{C.end}")
# else:
#     print(f"{C.red}---Their classifier was correct for more values of k---{C.end}")

def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0

def getInfoGain(data, classes, feature):
    # List values that feature can take
    values = []
    for datapoint in data:
        if datapoint[feature] not in values:
            values.append(datapoint[feature])
    
    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex = 0
    # Find where those values appear in data[feature] and the corresponding class
    for value in values:
        dataIndex = 0
        newClasses = []
        for datapoint in data:
            if datapoint[feature] == value:
                featureCounts[valueIndex] = featureCounts[valueIndex] + 1
                newClasses.append(classes[dataIndex])
            dataIndex = dataIndex + 1

        # Get the values in newClasses
        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass) == 0:
                classValues.append(aclass)

        classCounts = np.zeros(len(classValues))
        classIndex = 0
        for classValue in classValues:
            for aclass in classValue:
                if aclass == classValue:
                    classCounts[classIndex] = classCounts[classIndex] + 1
            classIndex = classIndex + 1

        for classIndex in range(len(classValues)):
            entropy[valueIndex] = entropy[valueIndex] + calc_entropy(
                float(classCounts[classIndex])/sum(classCounts))
        gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
        valueIndex = valueIndex + 1
    return gain

def findPath(graph, start, end, pathSoFar):
    pathSoFar = pathSoFar + [start]
    if start == end:
        return pathSoFar
    if start not in graph:
        return None
    for node in pathSoFar:
        if node not in pathSoFar:
            newpath = findPath(graph, node, end, pathSoFar)
            return newpath
    return None

def make_tree(data, classes, featureNames):
    default = classes[np.argmax(frequency)] # smells optional
    if nData == 0 or nFeatures == 0:
        # Have reached an empty branch
        return default
    elif classes.count(classes[0]) == nData:
        # Only 1 class remains
        return classes[0]
    else:
        # Choose best feature
        gain = np.zeros(nFeatures)
        for feature in range(nFeatures):
            g = getInfoGain(data, classes, feature)
            gain[feature] = totalEntropy - g
        bestFeature = np.argmax(gain)
        tree = {featureNames[bestFeature]:{}}
        # Find the datapoints with each feature value
        for datapoint in data:
            if datapoint[bestFeature] == value:
                if bestFeature == 0:
                    datapoint = datapoint[1:]
                    newNames = featureNames[1:]
                elif bestFeature == nFeatures:
                    datapoint = datapoint[:-1]
                    newNames = featureNames[:-1]
                else:
                    datapoint = datapoint[:bestFeature]
                    datapoint.extend(datapoint[bestFeature + 1:])
                    newNames = featureNames[:bestFeature]
                    newNames.extend(featureNames[:bestFeature + 1:])
                newData.append(datapoint)
                newClasses.append(classes[index])
            index = index + 1
        # Now to recurse to the next level
        subtree = make_tree(newData, newClasses, newNames)
        # And on returning, add the subtree on to the tree
        tree[featureNames[bestFeature]][value] = subtree
    return tree