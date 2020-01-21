import numpy as np
import statistics as stats

x = np.array([3, 6])
y = np.array([5, 2])
data = np.array([[2, 3], [3, 4], [5, 7], [2, 7], [3, 2], [1, 2], [9, 3], [4, 1]])
animals = ["dog", "cat", "bird", "fish", "fish", "dog", "cat", "dog"]

def findDistance(x, y):
    return np.sqrt(np.sum((x - y)**2))

print(findDistance(x, y))

distances = np.zeros(len(data))
for i in range(len(data)):
    distances[i] = findDistance(x, data[i])

array = np.argsort(distances)
for i in range(2):
    print(animals[array[i]])

print(stats.mode(animals))