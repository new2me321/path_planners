import numpy as np
import time
import math


class Djikstra:
    """
    Dijkstra's algorithm for finding the shortest path from start to goal.

    Parameters
    ----------
    start : array_like
        The starting node of the path.
    goal : array_like
        The goal node of the path.
    """

    def __init__(self, start, goal, tree, parent_pointers):

        self.start = start
        self.goal = goal
        self.tree = tree
        self.parent_pointers = parent_pointers

    def calc_distance(self, pt1, pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def get_path(self,):

        start_time = time.time()

        distances = {}  # holds the distances from the start to each point in the tree

        # Set the distance from the start to the goal to 0
        distances[tuple(self.goal)] = 0

        # Set the distance from the start to each point in the tree to infinity
        for point in self.tree:
            if not np.array_equal(point, self.goal):
                distances[tuple(point)] = float('inf')

        # contains all the points in the tree that were visited
        visited_nodes = [self.goal]

        while len(visited_nodes) < len(self.tree):
            # Find the point in the tree with the smallest distance from the start that has not yet been visited and add to the visited_nodes
            current_point = min([point for point in self.tree if tuple(point) not in [
                                tuple(p) for p in visited_nodes]], key=lambda x: distances[tuple(x)])
            visited_nodes.append(current_point)

            # Update the distances of all of the current point's neighbors to the start
            for neighbor in self.parent_pointers:
                if np.array_equal(self.parent_pointers[tuple(neighbor)], current_point):
                    distances[tuple(neighbor)] = min(distances[tuple(neighbor)], distances[tuple(
                        current_point)] + self.calc_distance(current_point, neighbor))

        shortest_path = [self.goal] # stores the shortest path

        # construct the shortest path from the goal to the start node using the parent pointers
        
        closest_goal_node = list(self.parent_pointers)[-1]
        current_point = closest_goal_node

        while current_point is not None and tuple(current_point) in self.parent_pointers:
            shortest_path.append(current_point)
            current_point = self.parent_pointers[tuple(current_point)]
        shortest_path.append(current_point)

        # Reverse the list to get the shortest path from the goal to the start node
        shortest_path = shortest_path[::-1]
        shortest_path = [np.array(point) for point in shortest_path]

        end_time = time.time()
        elapsed_time = end_time - start_time

        if len(shortest_path) > 1:
            print("Shortest path found!")
            print("Dijkstra's algorithm for finding the shortest path took {} seconds".format(
                elapsed_time))
        else:
            print("Shortest path not found!")

        return shortest_path
