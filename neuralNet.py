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
        self.height = numNodesPerLayer # For now each hidden layer has the same number of nodes
        self.depth = numHiddenLayers # Total layers = numHiddenLayers + 2 (one for inputs and outputs)
        self.bias = -1 # Bias to be added into each calculation
        # Initialize nodes structure
        nodes = [np.zeros(numClasses)] 
        for x in range(numHiddenLayers):
            nodes.append(np.zeros(numNodesPerLayer))
        nodes.append(np.zeros(numClasses))
        self.nodes = np.array(nodes)
        # Initialize weights structure
        self.weights = np.array([[[-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5]],
                                 [[-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5]],
                                 [[-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5], [-1, -0.5, 0.5, 1, 0.5]]])
    # This function applies the sigmond activation function to an array of nodes
    def sigmify(self, array):
        return np.array(list(map(lambda x: 1 / (1 + math.pow(math.e, -1 * x)), array)))
    # Just a little randomization function to set the weights, for now goes between -3.0 to 3.0
    def randomizeWeights(self, size):
        array = []
        for x in range(size):
            array.append((random.random() - 0.5) * 6)
        return np.array(array)
    # The classification algorithm
    def classify(self, array):
        # Take in the data as an array
        self.nodes[0] = array
        # Each outputs for node sets are the sum of the weights times the inputs
        for nodeSet in range(len(self.nodes) - 1):
            self.nodes[nodeSet + 1] = self.sigmify(self.weights[nodeSet]
                .dot(np.append(self.nodes[nodeSet], self.bias).transpose()))
        # Return estimations on each classification
        return self.nodes

nn = NeuralNet(4, 4, 2)
print(nn.classify([1.4, 2.3, 5.0, 3.8]))
print(nn.randomizeWeights(5))

# This works without the loop
# nodes[0] = [1.4, 2.3, 5.0, 3.8]
# # First index of nodes needs to be calculated
# nodes[1] = sigmify(weights[0].dot(np.append(nodes[0], bias).transpose()))
# # Second index of nodes
# nodes[2] = sigmify(weights[1].dot(np.append(nodes[1], bias).transpose()))
# # Third index
# nodes[3] = sigmify(weights[2].dot(np.append(nodes[2], bias).transpose()))
# # Convert answers to guesses

# print(nodes)

# # Show the x, y, and names respectively
# print(iris.data)
# print(iris.target)
# print(iris.target_names)