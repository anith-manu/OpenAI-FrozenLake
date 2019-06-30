import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv
import os, sys
from helpers import *
import networkx as nx
from search import *


# Setup the parameters
problem_id = int(sys.argv[1])
reward_hole = 0.0
is_stochastic = True

# Generate the environment
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)


print(env.desc)

state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

maze_map = UndirectedGraph(state_space_actions)

# initialise a graph
G = nx.Graph()

node_labels = dict()
node_colors = dict()
for n, p in state_space_locations.items():
    G.add_node(n)            # add nodes from locations
    node_labels[n] = n       # add nodes to node_labels
    node_colors[n] = "white" # node_colors to color nodes while exploring the map

# save the initial node colors to a dict for later use
initial_node_colors = dict(node_colors)

# positions for node labels
node_label_pos = {k:[v[0],v[1]-0.25] for k,v in state_space_locations.items()} # spec the position of the labels relative to the nodes

# used while labeling edges
edge_labels = dict()

# add edges between nodes in the map - UndirectedGraph defined in search.py
for node in maze_map.nodes():
    connections = maze_map.get(node)
    for connection in connections.keys():
        distance = connections[connection]
        G.add_edge(node, connection) # add edges to the graph
        edge_labels[(node, connection)] = distance # add distances to edge_labels

print("Done creating the graph object")



def my_best_first_graph_search(problem, f, initial_node_colors):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = dict(initial_node_colors)

    f = memoize(f, 'f')
    node = Node(problem.initial)

    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = PriorityQueue('min', f)
    frontier.append(node)

    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.pop()

        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None


def my_astar_search(initial_node_colors, problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')  # define the heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n), initial_node_colors)


graph_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)
    
print("Initial state: " + graph_problem.initial)
print("Goal state: " + graph_problem.goal)

all_node_colors = []
iterations, all_node_colors, node = my_astar_search(initial_node_colors, problem=graph_problem, h=None)
print(iterations)

# -- Trace the solution --#
solution_path = [node]
cnode = node.parent
solution_path.append(cnode)

while cnode.state != state_initial_id:
    cnode = cnode.parent
    solution_path.append(cnode)

print("----------------------------------------")
print("Identified goal state:" + str(solution_path[0]))
print("----------------------------------------")
print("Solution trace:" + str(solution_path))
print("----------------------------------------")
print("Iterations:" + str(iterations))
print()


# Print output to file
filename ="Simple_Agent_Output " + str(problem_id)

outputGoals = "Identified goal state:" + str(solution_path[0])  + "\n \nSolution trace: " + str(solution_path) + "\n \nIterations: " + str(iterations)

with open(filename + ".txt", "w") as file:
    if(len(solution_path) > 0):
        file.write(str(outputGoals))
    else:
        file.write(str(outputNone))


def return_iterations():
    return iterations
