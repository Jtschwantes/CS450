from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np
import math
import random

class NeuralNet:
    ############## INITIATION FUNCTIONS ##############
    # Constructor
    def __init__(self, numInputs, numNodesPerLayer, numHiddenLayers = 0):
        # Various initializations
        self.classes = numInputs # Number of inputs (TODO: Separate input and output numbers)
        self.height = numNodesPerLayer # For now each hidden layer has the same number of nodes
        self.depth = numHiddenLayers + 1 # Total layers = numHiddenLayers + 2 (one for inputs and outputs)
        self.bias = -1 # Bias to be added into each calculation
        self.learnRate = 0.1
        self.nodes = self.initNodes()
        self.weights = self.initWeights()
    # Initialize nodes
    def initNodes(self):
        # Initialize with the number of input classes
        nodes = [np.zeros(self.classes)]
        # For each hidden layer, add more empty nodes
        for x in range(self.depth - 1):
            nodes.append(np.zeros(self.height))
        # Add output layer (TODO: Make this support a unique number)
        nodes.append(np.zeros(self.classes))
        return np.array(nodes)
    # Just a little randomization function to set the weights, for now goes between -3.0 to 3.0
    def initWeights(self):
        temp = []
        temp2 = []
        array = []
        # Make tripple array of arrays
        #   Weight[layer][node][weight]
        for x in range(self.depth):
            for y in range(self.height):
                for z in range(self.height + 1):
                    temp.append((random.random() - 0.5) * 6)
                temp2.append(temp)
                temp = []
            array.append(temp2)
            temp2 = []
        return np.array(array)
    
    ################## UTILITY FUNCTIONS ####################
    # This function applies the sigmond activation function to an array of nodes
    def sigmify(self, array):
        return np.array(list(map(lambda x: 1 / (1 + math.pow(math.e, -1 * x)), array)))
    # Returns output array of error values given a target value
    def outputErr(self, arr, idx):
        # Make fully empty array except with a '1' for correct value
        targArr = np.zeros(self.classes)
        arr[idx] = 1
        newArr = []
        # For whole array, return a(1-a)(a-t)
        for i in range(len(arr)):
            newArr.append(arr[i] * (1 - arr[i]) * (arr[i] - targArr[i]))
        return newArr
    # Returns an array of error terms for a give error term in the array 
    #    ahead of it. Used to find error term for hidden layer nodes
    def hiddenErr(self, errors, idx):
        newArr = []
        # Just grapping the appropriate weights and nodes for simplicity
        nodes = self.nodes[idx]
        weights = self.weights[idx - 1]
        for i in range(len(nodes)):
            # First, sum the node errors and the weights
            mySum = 0
            for j in range(len(weights - 1)):
                mySum = mySum + (weights[i][j + 1] * errors[j])
            # Then for whole array, return a(1-a)(sum: weights(errors))
            newArr.append(nodes[i] * (1 - nodes[i]) * mySum)
        return newArr

    ################ PUBLIC FUNCTIONS ##########################
    # The classification algorithm:
    #   Takes a set of input nodes and feeds it forward
    def classify(self, array):
        # Take in the data as an array
        self.nodes[0] = array
        # Each outputs for node sets are the sum of the weights times the inputs
        for nodeSet in range(len(self.nodes) - 1):
            # Applies the sigmond activation to the weights dotted with the 
            #   inputs and biases, transposed
            self.nodes[nodeSet + 1] = self.sigmify(self.weights[nodeSet]
                .dot(np.append(self.nodes[nodeSet], self.bias).transpose()))
        # Return estimations on each classification
        return self.nodes[-1]
    # Returns a set of nodes that represents the error term for each node
    def errorize(self, idx):
        errors = []
        temp = []
        # For output layer, apply output error function
        for x in range(self.depth - 1):
            errors.append(np.zeros(self.height))
        # Save temporary array with error values for increased layer,
        #   used for hidden layer error terms
        temp = self.outputErr(self.nodes[-1], idx)
        errors.append(temp)
        # For hidden layers, apply hidden error function
        for i in reversed(range(self.depth - 1)):
            temp = self.hiddenErr(temp, i + 1)
            errors[i] = temp
        return errors
    # Back Propogation: takes error values and changes the weights
    def learn(self, errors):
        newArr = []
        # For each layer
        for index in range(len(self.weights)):
            # For each node
            for idx in range(len(self.weights[index])):
                # For each weight
                for i in range(len(self.weights[index][idx]) - 1):
                    # weight = weight - LR * Err(layer1) * a(layer2)
                    self.weights[index][idx][i + 1] = (self.weights[index][idx][i + 1] - 
                        ((self.learnRate * errors[index][i]) * self.nodes[index][i]))

nn = NeuralNet(4, 4, 2)
print(nn.classify([1.4, 2.3, 5.0, 3.8]))
nn.learn(nn.errorize(1))