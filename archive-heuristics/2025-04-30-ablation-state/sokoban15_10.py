from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Utility function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Utility function to match a PDDL fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS function to find shortest paths in an unweighted graph
def bfs(start_node, graph):
    """
    Performs a Breadth-First Search to find shortest paths from a start node
    in an unweighted graph.

    Args:
        start_node: The node to start the search from.
        graph: An adjacency list representation of the graph {node: [neighbors]}.

    Returns:
        A dictionary mapping each reachable node to its shortest distance from the start_node.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in distances: # Handle case where start_node is not in the graph nodes
        return {} # Cannot reach anything if start node is not in graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is a valid key in the graph
        if current_node in graph:
            for neighbor in graph[current_node]:
                if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components for each box that is not yet at its goal:
    1. The shortest path distance from the box's current location to its goal location (estimating the minimum number of pushes required).
    2. The shortest path distance from the robot's current location to the nearest location adjacent to the box (estimating the robot movement needed to get into a position to push the box).

    # Assumptions
    - The grid is represented by locations connected by `adjacent` predicates.
    - Shortest path distances are calculated on the static grid graph, ignoring dynamic obstacles (other boxes, robot) and `clear` predicates. This is a simplification for efficiency and results in a non-admissible heuristic.
    - Each box has a unique goal location. (Based on example instances).

    # Heuristic Initialization
    - Builds the location graph from `adjacent` facts in the static information.
    - Determines the goal location for each box from the goal conditions.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Initialize the total heuristic cost to 0.
    4. For each box specified in the goal conditions:
        a. Get the box's current location and its goal location.
        b. If the box is not at its goal location:
            i. Calculate the shortest path distance from the box's current location to its goal location using the precomputed distances. Add this distance to the total cost. This estimates the pushes needed for this box.
            ii. Find all locations adjacent to the box's current location using the precomputed graph.
            iii. Calculate the shortest path distance from the robot's current location to each of these adjacent locations.
            iv. Find the minimum of these distances. Add this minimum distance to the total cost. This estimates the robot movement needed to get to a position from which it can push the box.
            v. If either the box goal is unreachable or no location adjacent to the box is reachable by the robot, return infinity (indicating a likely dead end or unsolvable state).
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, identifying
        goal locations, and precomputing all-pairs shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Collect all unique locations mentioned in the problem
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["adjacent", "at-robot", "at", "clear"]:
                 # Add all arguments that are not directions
                 all_locations.update(p for p in parts[1:] if p not in ["up", "down", "left", "right"])

        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] in ["at-robot", "at", "clear"]:
                  all_locations.update(parts[1:])

        for goal in self.goals:
             # Handle (and (...)) structure
             if get_parts(goal)[0] == "and":
                  for sub_goal_str in get_parts(goal)[1:]:
                       sub_goal_parts = get_parts(sub_goal_str)
                       if sub_goal_parts and sub_goal_parts[0] == "at":
                            all_locations.add(sub_goal_parts[2]) # Add goal location
             elif get_parts(goal)[0] == "at":
                  all_locations.add(get_parts(goal)[2]) # Add goal location


        # Build the location graph from adjacent facts
        self.location_graph = {loc: [] for loc in all_locations} # Initialize graph with all locations
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                # Add edge loc1 -> loc2. Since PDDL lists both directions, this creates a bidirectional graph.
                if loc1 in self.location_graph: # Ensure loc1 is a known location
                    self.location_graph[loc1].append(loc2)


        # Precompute all-pairs shortest paths
        self.all_pairs_distances = {}
        # Iterate through all collected locations to ensure BFS is run from all potential start nodes
        for location in all_locations:
             # Ensure location is a key in the graph dictionary before running BFS from it
             if location not in self.location_graph:
                 self.location_graph[location] = [] # Add it as a key with no outgoing edges if it wasn't already
             self.all_pairs_distances[location] = bfs(location, self.location_graph)


        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            # Handle (and (...)) structure
            if get_parts(goal)[0] == "and":
                 for sub_goal_str in get_parts(goal)[1:]:
                      sub_goal_parts = get_parts(sub_goal_str)
                      if sub_goal_parts and sub_goal_parts[0] == "at":
                           _, box, location = sub_goal_parts
                           self.goal_locations[box] = location
            elif get_parts(goal)[0] == "at":
                 _, box, location = get_parts(goal)
                 self.goal_locations[box] = location


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current robot location
        current_robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                _, current_robot_loc = get_parts(fact)
                break # Assuming only one robot

        # If robot location is not found, something is wrong
        if current_robot_loc is None:
             return float('inf') # Should not happen in valid states

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                current_box_locations[box] = loc

        total_cost = 0  # Initialize action cost counter.

        # Sum costs for each box not at its goal
        for box, goal_location in self.goal_locations.items():
            current_box_location = current_box_locations.get(box)

            # If a box from the goal is not found in the current state,
            # something is wrong or it's an unreachable state.
            if current_box_location is None:
                 return float('inf') # Should not happen in valid states

            if current_box_location != goal_location:
                # Cost 1: Distance from box to its goal
                # Check if goal_location is reachable from current_box_location
                box_dist = self.all_pairs_distances.get(current_box_location, {}).get(goal_location, float('inf'))
                if box_dist == float('inf'):
                     # If a box goal is unreachable, this state is likely a dead end or unsolvable.
                     return float('inf')
                total_cost += box_dist

                # Cost 2: Distance from robot to the nearest location adjacent to the box
                min_robot_dist_to_adjacent = float('inf')
                # Get neighbors of the box's current location from the graph
                box_neighbors = self.location_graph.get(current_box_location, [])

                # Calculate distance from robot to each neighbor and find the minimum
                robot_distances_from_current_loc = self.all_pairs_distances.get(current_robot_loc, {})
                for neighbor_loc in box_neighbors:
                    robot_dist_to_neighbor = robot_distances_from_current_loc.get(neighbor_loc, float('inf'))
                    min_robot_dist_to_adjacent = min(min_robot_dist_to_adjacent, robot_dist_to_neighbor)

                if min_robot_dist_to_adjacent == float('inf'):
                    # If the robot cannot reach any location adjacent to the box, it cannot push it.
                    return float('inf')
                total_cost += min_robot_dist_to_adjacent

        return total_cost
