import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from djikstra import Djikstra


class RRT:
    """
    Parameters
    ----------
    start : starting point
    goal : goal point
    max_step_size : maximum distance between a tree node to a new point
    max_iter : maximum number of iterations
    goal_tolerance : how close to the goal

    """

    def __init__(self, start, goal,   map_size, max_iter=10000, max_step_size=0.1, goal_tolerance=0.5):
        self.start = start  # starting point
        self.goal = goal  # goal point
        self.goal_tolerance = goal_tolerance  # tolerance
        self.max_step_size = max_step_size  # distance between a node and a new point
        self.max_iter = max_iter
        self.map_width = map_size[0]
        self.map_height = map_size[1]
        self.parent_pointers = {}

        print("RRT initializating...")

    def calc_distance(self, pt1, pt2):
        return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

    def calc_direction(self, pt1, pt2):
        distance = self.calc_distance(pt1, pt2)
        return (pt1-pt2)/distance

    def findNearestNode(self, point, tree):
        nearest_node = tree[0]
        nearest_distance = self.calc_distance(point, nearest_node)
        for node in tree:
            distance = self.calc_distance(point, node)
            if distance < nearest_distance:
                nearest_node = node
                nearest_distance = distance

        return nearest_node

    def generateNewNode(self, point, nearest_node):

        # compute distance between a new point and nearest_node of a tree
        distance = self.calc_distance(point, nearest_node)

        if distance < self.max_step_size:
            return point
        else:
            new_node = self.newNodeSelector(point, nearest_node)

            # if new_node hits an obstacle:
            # return None

        return new_node

    def newNodeSelector(self, point, nearest_node):
        """
        Selects a new node 
        """
        # get the vector direction
        direction = self.calc_direction(point, nearest_node)

        new_node = nearest_node + (direction*self.max_step_size)

        return new_node

    def generateRandomPoint(self):
        x = np.random.uniform(0, self.map_width)
        y = np.random.uniform(0, self.map_height)

        return np.array([x, y])

    def find_path(self):
        """
        Finds the path to the goal point
        """
        found = False
        tree = [self.start]
        elapsed_iters = 0

        start_time = time.time()
        for i in range(self.max_iter):
            elapsed_iters += 1
            random_point = self.generateRandomPoint()
            nearest_node = self.findNearestNode(
                random_point, tree)
            new_node = self.generateNewNode(random_point, nearest_node)

            if new_node is not None:
                tree.append(new_node)
                self.parent_pointers[tuple(
                    new_node)] = nearest_node[0], nearest_node[1]

            # check if the node is already near the goal point
            if self.calc_distance(new_node, self.goal) <= self.goal_tolerance:
                found = True
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        if found:
            print("RRT path found in {} seconds".format(elapsed_time))
            print("Iterations taken:", elapsed_iters)

        return found if found is False else tree

    def get_path(self, use_djikstra):
        tree = self.find_path()

        if tree is not False:

            if use_djikstra == True:
                djikstra = Djikstra(self.start, self.goal, np.array(
                    tree), self.parent_pointers)
                path = djikstra.get_path()
            else:
                tree_node_distances = [self.calc_distance(
                    node, self.goal) for node in tree]

                # sort the tree nodes by distance
                sorted_trees = [node for _, node in sorted(
                    zip(tree_node_distances, tree))]
                # print("Sorted trees", sorted_trees)
                path = []  # stores the path to the goal

                next_node = self.goal
                dist_limit = self.calc_distance(self.start, self.goal)
                print("Limit", dist_limit)

                for node in sorted_trees:
                    node_to_goal = self.calc_distance(node, self.goal)
                    nearest_dist = self.calc_distance(node, next_node)

                    if (node_to_goal > dist_limit):#(nearest_dist <= self.goal_tolerance):
                        next_node = node
                        break
                    else:   
                        path.append(node)

                print("Path:", len(path))

            return np.array(tree), np.array(path)
        else:
            raise ValueError("Could not find path to goal")

    def visualize(self):
        tree, path = self.get_path(use_djikstra=False)

        # Extract the x and y coordinates of the points in the path
        nodes_x, nodes_y = np.split(tree, 2, axis=1)

        # Set up the figure and axis
        fig, ax = plt.subplots()

        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)

        # draw points
        scatter = ax.scatter(nodes_x, nodes_y, s=1.5, c='black', label='nodes')
        ax.scatter(self.start[0], self.start[1], s=20,
                   c='red', marker='*', label='Start')
        ax.scatter(self.goal[0], self.goal[1], s=20,
                   c='green', marker='*', label='Goal')

        # Function to update the plot at each frame
        def update(num):

            # Update the x and y data of the line plot
            scatter.set_offsets(tree[:num])
            ax.set_title("RRT iterations: " + str(num))
            if len(nodes_x)-1 == num:
                ax.scatter(nodes_x[-1], nodes_y[-1], s=5,
                           c='gold', label='last node')
                ax.plot(path[:, 0], path[:, 1], label='shortest path')
                ax.legend(loc='upper right')

            return scatter,

        # Create the animation
        animation = FuncAnimation(fig, update, frames=len(
            nodes_x), interval=2, repeat=False)

        ax.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    # np.random.seed(0)  # set seed to repeat same randomness
    start = np.array([5, 70])
    goal = np.array([79, 30])
    max_iter = 1000
    max_step_size = 5
    goal_tolerance = 2*max_step_size
    map_size = (100, 100)

    rrt = RRT(start=start, goal=goal, max_iter=max_iter,
              max_step_size=max_step_size, map_size=map_size, goal_tolerance=goal_tolerance)

    rrt.visualize()