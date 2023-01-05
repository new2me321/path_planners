import numpy as np
import time
import math

class Astar:
    def __init__(self, start, goal, tree, parent_pointers):
        self.start = start
        self.goal = goal
        self.tree = tree
        self.parent_pointers = parent_pointers
        self.path = []
