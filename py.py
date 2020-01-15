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
class MyClassifier:
    def __init__(self):
        self.xTrain = []
        self.yTrain = []
    def fit(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain
    def predict(self, xTest):
        yTest = np.zeros(len(xTest), dtype=int)
        for i in range(len(xTest)):
            yTest[i] = 0
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