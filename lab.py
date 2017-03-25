import json
import copy
import math

import numpy as np  # contains helpful math functions like numpy.exp()
# import numpy.random as random  # see numpy.random module
import random  # alternative to numpy.random module

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NEIGHBOR_METHOD_SWAP = "closestswaped"
NEIGHBOR_METHOD_RANDOM = "random"

DISTANCE_EUCLIDIAN = "euclidian"
DISTANCE_MANHATTAN = "manhattan"

neighborMethod = NEIGHBOR_METHOD_RANDOM
distanceMethod = DISTANCE_EUCLIDIAN

map = mpimg.imread("map.png")  # US States & Capitals map

# List of 30 US state capitals and corresponding coordinates on the map
with open('capitals.json', 'r') as capitals_file:
    capitals = json.load(capitals_file)

capitals_list = list(capitals.items())

def show_path(path, starting_city, w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    x, y = list(zip(*path))
    _, (x0, y0) = starting_city
    plt.imshow(map)
    plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    plt.show()

def simulated_annealing(problem, schedule):
    
    current = problem
    
    for t in range(1, 2000000):
        T = schedule(t)
        
        if T <= 1e-10:
            return current
        
        possible = problem.successors()
        random.shuffle(possible)
        next = possible[0]
        
        distance = next.get_value() - current.get_value()
        
        if distance > 0: 
            current = next
        else: 
            probability = np.exp(distance/T)
            if random.random() < probability:
                current = next
    
    return current

class TravelingSalesmanProblem:
    
    def __init__(self, cities):
        self.path = copy.deepcopy(cities)
    
    def copy(self):
        """Return a copy of the current board state."""
        new_tsp = TravelingSalesmanProblem(self.path)
        return new_tsp
    
    @property
    def names(self):
        names, _ = zip(*self.path)
        return names
    
    @property
    def coords(self):
        _, coords = zip(*self.path)
        return coords
    
    def successors(self):
        
        if neighborMethod == NEIGHBOR_METHOD_SWAP:
            return self.__successors_closest_swapped()
        else:
            return self.__successors_random_swap()

    def __successors_random_swap(self):
        states = []
        
        newState = self.copy()

        random.shuffle(newState.path)

        states.append(newState)
        
        return states

    def __successors_closest_swapped(self):
        states = []
        
        for i in range(0, len(self.path)):
            newState = self.copy()
            switchIndex = i + 1
            if i == len(self.path) - 1:
                switchIndex = 0
            
            t = newState.path[i]
            newState.path[i] = newState.path[switchIndex]
            newState.path[switchIndex] = t
            
            states.append(newState)
        
        return states

    def get_value(self):
        
        totalDistance = 0
        for i in range(0, len(self.coords)):
            p1 = self.coords[i]
            
            if i == len(self.coords) - 1:
                p2 = self.coords[0]
            else:
                p2 = self.coords[i+1]
            
            if distanceMethod == DISTANCE_EUCLIDIAN:
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
            else:
                distance = abs(p1[0]-p2[0]) + abs(p1[1] - p2[1])
            
            totalDistance += distance
            
        return -1 * totalDistance

alpha = 0.95
temperature=1e4

def schedule(time):
    return alpha ** time * temperature

def solve(tsp):
    return simulated_annealing(capitals_tsp, schedule)

num_cities = 30
alpha = 0.95
temperature=1e6
trials = 30

smallest = 200000
smallestResult = None

capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
starting_city = capitals_list[0]
print("Initial path value: {:.2f}".format(-capitals_tsp.get_value()))
# print(capitals_list[:num_cities])  # The start/end point is indicated with a yellow star
print(capitals_tsp.names)  # The start/end point is indicated with a yellow star
# show_path(capitals_tsp.coords, starting_city)

for i in range(0,trials):
    capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities])
    result = solve(capitals_tsp)
    value = -result.get_value()
    if value < smallest:
        smallest = value
        smallestResult = result

    print("Final path length: {:.2f}".format(value),"index",i)
    # print(result.names)

print("smallest distance",smallest)
# show_path(smallestResult.coords, starting_city)