from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# # Show the x, y, and names respectively
# print(iris.data)
# print(iris.target)
# print(iris.target_names)

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

xTrain, xTest, yTrain, yTest = train_test_split(
    iris.data, iris.target, test_size = 0.3)

classifier = GaussianNB()
classifier.fit(xTrain, yTrain)
results = classifier.predict(xTest)

print(f"Expected classifications: {results}")
print(f"  Actual classifications: {yTest}")
print()

myClassifier = MyClassifier()
myClassifier.fit(xTrain, yTrain)
results = myClassifier.predict(xTest)

print(f"Expected classifications: {results}")
print(f"  Actual classifications: {yTest}")
print()