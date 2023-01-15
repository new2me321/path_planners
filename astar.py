import numpy as np
import time
import math
import ast

class Astar:
    """
    A-star algorithm for finding the shortest path from start to goal.

    Parameters
    ----------
    start : array_like
        The starting node of the path.
    goal : array_like
        The goal node of the path.
    """

    def __init__(self, start, goal,  graph):
        self.start = start
        self.goal = goal
        self.graph = graph
        self.alpha = 1  # the heuristic term

    def calc_distance(self, pt1, pt2):

        # print("Calc distance", pt1, pt2)
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def compute_heuristic_cost(self, node, goal):
        """
        Compute the cost from the current node to the goal
        """
        heuristic = self.alpha * \
            math.sqrt((node[0]-goal[0])**2 + (node[1]-goal[1])**2)
        return heuristic

    def get_path(self,):
        """
        The main function of A* algorithm.

        F = G + H
        F:= total cost
        G:= the cost from the current node to the start
        H:= the estimate cost from the current node to the goal
        """
        start_time = time.time()

        # initialize both open and closed lists
        # add start node to open list

        open_list = [tuple(self.start)]
        closed_list = []         # contains all the points in the graph that were visited

        # dictionaries to store F and G scores
        F_scores = {}  # total cost from start to goal
        G_scores = {}  # total cost from start node to current node

        # store scores for start node
        G_scores[tuple(self.start)] = 0
        F_scores[tuple(self.start)] = self.compute_heuristic_cost(
            self.start, self.goal)

        # path from goal to start
        path = {}
        path[tuple(self.start)] = None

        shortest_path = []

        while len(open_list) != 0:
            # find the highest priority node i.e node with lowest F score
            current_idx = np.argmin([F_scores[node] for node in open_list])
            current_node = open_list[current_idx]

            # add the current node closed list (i.e visited_nodes)
            closed_list.append(current_node)

            # remove the current node from the open list
            open_list = list(filter(lambda x: x != current_node, open_list))

            if current_node == tuple(self.goal):
                shortest_path = self.reconstruct_path(path)
                break
            try:
                next_nodes = self.graph[current_node]
            except KeyError:
                continue
            else:
                for neighbor in next_nodes:
                    if neighbor in closed_list:
                        continue

                    new_g_score = G_scores[current_node] + self.calc_distance(current_node, neighbor)

                    if neighbor not in open_list:
                        open_list.append(neighbor)

                    if neighbor not in F_scores or new_g_score < G_scores[neighbor]:
                        G_scores[neighbor] = new_g_score
                        F_scores[neighbor] = G_scores[neighbor] + self.compute_heuristic_cost(neighbor, self.goal)
                        path[tuple(neighbor)] = current_node

        end_time = time.time()
        elapsed_time = end_time - start_time

        if len(shortest_path) > 1:
            print("Shortest path found!")
            print("A-star algorithm for finding the shortest path took {} seconds".format(
                elapsed_time))
        else:
            # print("Shortest path not found!")
            raise ValueError("Shortest path not found!")

        return shortest_path

    def reconstruct_path(self, visited):
        shortest_path = []
        current_node = tuple(self.goal)

        while current_node is not None and tuple(current_node) in self.graph:
            shortest_path.append(current_node)
            current_node = visited[tuple(current_node)]

        # Reverse the list to get the shortest path from the goal to the start node
        shortest_path = shortest_path[::-1]
        shortest_path = [np.array(point) for point in shortest_path]

        return shortest_path


if __name__ == "__main__":
    with open('graph_data.txt', 'r') as f:
        graph = ast.literal_eval(f.read())

    start = np.array([75, 70])
    goal = np.array([5, 20])

    astar = Astar(start, goal, graph)
    astar.get_path()
