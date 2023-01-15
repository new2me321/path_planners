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
    graph : a dictionary of tuples where the values are the list of nodes(vertices). 
        For eg. 
        G = {(1, 2) : [(3,5), (2,4), (1,3)],
            (1, 3) : [(3,5), (2,2)],
            (2, 2) : [(3,5), (2,6), (3,3)]
            }
    """

    def __init__(self, start, goal, graph):

        self.start = start
        self.goal = goal
        self.graph = graph

    def calc_distance(self, pt1, pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def get_path(self,):

        start_time = time.time()

        distances = {}  # holds the distances from the start to each point in the graph

        # Set the distance from the start to each point in the graph to infinity
        for point in self.graph:
            if not np.array_equal(point, self.goal):
                distances[tuple(point)] = float('inf')

        # Set the distance from the start to 0  and the distance to goal to infinity
        distances[tuple(self.start)] = 0
        distances[tuple(self.goal)] = float('inf')

        # contains all the points in the graph that were visited
        visited_nodes = []

        # path from goal to start
        path = {}
        path[tuple(self.start)] = None

        shortest_path = []
        while len(visited_nodes) < len(self.graph):

            # Find the point in the graph with the smallest distance from the start that has not yet been visited and add to the visited_nodes
            current_point = min([point for point in self.graph if tuple(point) not in [
                                tuple(p) for p in visited_nodes]], key=lambda x: distances[tuple(x)])
            visited_nodes.append(current_point)

            if current_point == tuple(self.goal):
                shortest_path = self.reconstruct_path(path)
                break

            for neighbor in self.graph[current_point]:

                new_cost = distances[tuple(
                    current_point)] + self.calc_distance(current_point, neighbor)

                if neighbor not in distances or new_cost < distances[neighbor]:
                    distances[tuple(neighbor)] = new_cost
                    path[tuple(neighbor)] = current_point

        end_time = time.time()
        elapsed_time = end_time - start_time

        if len(shortest_path) > 1:
            print("Shortest path found!")
            print("Dijkstra's algorithm for finding the shortest path took {} seconds".format(
                elapsed_time))
        else:
            # print("Shortest path not found!")
            raise ValueError("Shortest path not found!")

        return shortest_path

    def reconstruct_path(self, visited):
        shortest_path = []
        current_point = tuple(self.goal)

        while current_point is not None and tuple(current_point) in self.graph:
            shortest_path.append(current_point)
            current_point = visited[tuple(current_point)]

        # Reverse the list to get the shortest path from the goal to the start node
        shortest_path = shortest_path[::-1]
        shortest_path = [np.array(point) for point in shortest_path]

        return shortest_path
