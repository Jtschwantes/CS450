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
            ctr = ctr + 1
    return ctr / len(exp)

# Tests the data multiple times and takes an average (displays it colorfully!)
# K testing
kLow = 1
kHigh = 22
kStep = 2
# Number of tests per K
numTests = 30

print()
print(f"{C.purple}Testing custom kNN{C.end}")
score = 0 # To see who wins!
# Loop through each K
for a in range(kLow, kHigh, kStep):
    print(f"{C.blue}Testing k of {a} {C.red}({numTests} Tests){C.end}")
    myAccuracy = []
    theirAccuracy = []
    for b in range(numTests):
        # Reset data
        xTrain, xTest, yTrain, yTest = train_test_split(
            iris.data, iris.target, test_size = 0.3)
        # My implementation
        knn = kNN()
        knn.fit(xTrain, yTrain)
        results = knn.predict(xTest, 5)
        my_accuracy = calcAvg(results, yTest)
        myAccuracy.append(my_accuracy)
        # Their implementation
        classifier = KNeighborsClassifier(n_neighbors=a)
        classifier.fit(xTrain, yTrain)
        predictions = classifier.predict(xTest)
        their_accuracy = calcAvg(predictions, yTest)
        theirAccuracy.append(their_accuracy)

    # Results print out
    me = (sum(myAccuracy) / numTests)
    them = (sum(theirAccuracy) / numTests)
    print(f"{C.purple}     My Accuracy: {C.yellow}{round(me * 100, 2)}{C.end}")
    print(f"{C.purple}  Their Accuracy: {C.yellow}{round(them * 100, 2)}{C.end}")
    if me > them:
        print(f"{C.green}     Mine won!{C.end}")
        score = score + 1
    elif me == them:
        print(f"{C.yellow}     We tied{C.end}")
    else:
        print(f"{C.red}     Mine lost{C.end}")
        score = score - 1

# Prints the final decision
print()
if score > 0:
    print(f"{C.green}---My classifier was correct for more values of k---{C.end}")
elif score == 0:
    print(f"{C.yellow}---Our classifiers tied---{C.end}")
else:
    print(f"{C.red}---Their classifier was correct for more values of k---{C.end}")