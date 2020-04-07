from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np
import math
import random

iris = datasets.load_iris()

class NeuralNet:
    def __init__(self, numClasses, numNodesPerLayer, numHiddenLayers = 0):
        # Various initializations
        self.classes = numClasses
        self.height = numNodesPerLayer # For now each hidden layer has the same number of nodes
        self.depth = numHiddenLayers + 1 # Total layers = numHiddenLayers + 2 (one for inputs and outputs)
        self.bias = -1 # Bias to be added into each calculation
        self.learnRate = 0.1
        # Initialize nodes structure
        nodes = [np.zeros(numClasses)] 
        for x in range(numHiddenLayers):
            nodes.append(np.zeros(numNodesPerLayer))
        nodes.append(np.zeros(numClasses))
        self.nodes = np.array(nodes)
        # Initialize weights structure
        self.weights = self.initWeights()
    # Just a little randomization function to set the weights, for now goes between -3.0 to 3.0
    def initWeights(self):
        temp = []
        temp2 = []
        array = []
        for x in range(self.depth):
            for y in range(self.height):
                for z in range(self.height + 1):
                    temp.append((random.random() - 0.5) * 6)
                temp2.append(temp)
                temp = []
            array.append(temp2)
            temp2 = []
        return np.array(array)
    # This function applies the sigmond activation function to an array of nodes
    def sigmify(self, array):
        return np.array(list(map(lambda x: 1 / (1 + math.pow(math.e, -1 * x)), array)))
    # Returns an array of error values given a target value
    def outputErr(self, arr, idx):
        targArr = np.zeros(self.classes)
        arr[idx] = 1
        newArr = []
        for i in range(len(arr)):
            newArr.append(arr[i] * (1 - arr[i]) * (arr[i] - targArr[i]))
        return newArr
    # Returns an array of error terms for a given
    #   error term in the array ahead of it
    def hiddenErr(self, errors, idx):
        newArr = []
        nodes = self.nodes[idx]
        weights = self.weights[idx - 1]
        for i in range(len(nodes)):
            mySum = 0
            for j in range(len(weights - 1)):
                mySum = mySum + (weights[i][j + 1] * errors[j])
            newArr.append(nodes[i] * (1 - nodes[i]) * mySum)
        return newArr
    # The classification algorithm
    def classify(self, array):
        # Take in the data as an array
        self.nodes[0] = array
        # Each outputs for node sets are the sum of the weights times the inputs
        for nodeSet in range(len(self.nodes) - 1):
            self.nodes[nodeSet + 1] = self.sigmify(self.weights[nodeSet]
                .dot(np.append(self.nodes[nodeSet], self.bias).transpose()))
        # Return estimations on each classification
        return self.nodes[-1]
    def errorize(self, idx):
        errors = []
        temp = []
        for x in range(self.depth - 1):
            errors.append(np.zeros(self.height))
        temp = self.outputErr(self.nodes[-1], idx)
        errors.append(temp)
        for i in reversed(range(self.depth - 1)):
            temp = self.hiddenErr(temp, i + 1)
            errors[i] = temp
        return errors
    # Back Propogation
    def learn(self, errors):
        newArr = []
        for index in range(len(self.weights)):
            for idx in range(len(self.weights[index])):
                for i in range(len(self.weights[index][idx]) - 1):
                    # weight = weight - LR * Err(layer1) * a(layer2)
                    self.weights[index][idx][i + 1] = (self.weights[index][idx][i + 1] - 
                        ((self.learnRate * errors[index][i]) * self.nodes[index][i]))

nn = NeuralNet(4, 4, 2)
print(nn.classify([1.4, 2.3, 5.0, 3.8]))
errs = nn.errorize(1)
ws = nn.weights
nn.learn(errs)