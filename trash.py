# My Color class
class C:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    end = '\033[0m'

file = open("assets/car.csv", "r")
key = []
dic = {}

# Get number of columns
numCols = 0
for line in file:
    for value in line.split(","):
        numCols = numCols + 1
    break

# Prepare new dictionary
for x in range(numCols):
    dic[x] = []

# Get unique list of values
ctr = 0
for line in file:
    for value in line.split(","):
        datum = value.strip()
        if datum not in dic[ctr]:
            dic[ctr].append(datum)
        ctr = ctr + 1
        if ctr == numCols:
            ctr = 0
print(f"{C.yellow}{dic}")

# Prepare new dictionary of lookup values
lookup = {}
rlookup = {}
for x in range(numCols):
    lookup[x] = {}
    rlookup[x] = {}

# Prepare look up based off of values
counter = 0
counter2 = 0
for y in range(numCols):
    for val in dic[y]:
        lookup[counter2][val] = counter
        rlookup[counter2][counter] = val
        counter = counter + 1
    counter = 0
    counter2 = counter2 + 1
print(f"{C.blue}{lookup}")
print(f"{C.green}{rlookup}{C.end}")

# # Finally, prepare the data
# data = []
# counter3 = 0
# for line in open("assets/car.data", "r"):
#     newArr = []
#     for val in line.split(","):
#         val = val.strip()
#         newArr.append(lookup[counter3][val])
#         counter3 = counter3 + 1
#     data.append(newArr)
#     counter3 = 0