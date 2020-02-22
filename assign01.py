from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np

# # Show the x, y, and names respectively
# print(iris.data)
# print(iris.target)
# print(iris.target_names)
iris = datasets.load_iris()

# My classifier class
class kNN:
    def __init__(self):
        self.xTrain = []
        self.yTrain = []
    # Fit doesn't really do much
    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain
    # Finds the K nearest neighbors
    def predict(self, xTest, k):
        yTest = [] # Results array
        for test in xTest: # Loop through each test point
            results = [] # Temporary array to store yTrain results
            # First, calculate distances to training points
            step = list(map(lambda x:(x - test)**2, self.xTrain))
            # Then, sum them into one value for each element
            step2 = list(map(np.sum, step))
            # Grab the indexes of the smallest distances
            indexes = np.argsort(step2)
            # Grab the smallest k distances
            for j in range(k):
                results.append(self.yTrain[indexes[j]])
            # Classify by getting the mode
            yTest.append(stats.mode(results))
        return yTest

# Function to get accuracy
# Formula for accuracy: TP + TN / Everything
def calcAvg(exp, act):
    ctr = 0
    for i in range(len(exp)):
        if exp[i] == act[i]:
            ctr += 1
    return ctr / len(exp)

xTrain, xTest, yTrain, yTest = train_test_split(
    iris.data, iris.target, test_size = 0.3)

classifier = GaussianNB()
classifier.fit(xTrain, yTrain)
results = classifier.predict(xTest)
accuracy = calcAvg(results, yTest)

print(f"Expected classifications: {results}")
print(f"  Actual classifications: {yTest}")
print(f"                Accuracy: {round(accuracy * 100, 2)}")
print()

myClassifier = MyClassifier()
myClassifier.fit(xTrain, yTrain)
results2 = myClassifier.predict(xTest)
accuracy = calcAvg(results2, yTest)

print(f"Expected classifications: {results2}")
print(f"  Actual classifications: {yTest}")
print(f"                Accuracy: {round(accuracy * 100, 2)}")
print()