from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_all_locations(static_facts, initial_state, goals):
    """Collect all unique locations mentioned in static facts, initial state, and goals."""
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            locations.add(parts[1])
            locations.add(parts[2])
    for fact in initial_state:
         parts = get_parts(fact)
         if parts[0] == 'at-robot':
             locations.add(parts[1])
         elif parts[0] == 'at':
             locations.add(parts[2])
    for goal in goals:
         parts = get_parts(goal)
         if parts[0] == 'at':
             locations.add(parts[2])
    return list(locations)

def build_graph(static_facts):
    """Build an undirected graph from adjacent facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            loc1, loc2 = parts[1], parts[2]
            if loc1 not in graph:
                graph[loc1] = []
            if loc2 not in graph:
                graph[loc2] = []
            # Add bidirectional edges assuming adjacency is symmetric for robot movement
            graph[loc1].append(loc2)
            graph[loc2].append(loc1)
    # Remove duplicates from adjacency lists
    for loc in graph:
        graph[loc] = list(set(graph[loc]))
    return graph

def bfs_distance(start_loc, graph):
    """Compute shortest path distances from start_loc to all reachable locations."""
    distances = {loc: math.inf for loc in graph}
    if start_loc in graph: # Ensure start_loc is in the graph
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            if current_loc in graph: # Should always be true if queue populated from graph keys
                for neighbor in graph[current_loc]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
    return distances

def precompute_distances(graph, all_locations):
    """Precompute all-pairs shortest path distances."""
    all_distances = {}
    # Need to include locations that might be isolated (not in graph keys)
    # BFS from a location not in the graph will return inf for all others, which is correct.
    # We need to ensure BFS is run for *all* relevant locations (from initial, goals, static).
    for start_loc in all_locations:
         all_distances[start_loc] = bfs_distance(start_loc, graph)
    return all_distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box not yet at its goal:
    1. The shortest path distance for the box from its current location to its goal location (minimum pushes required).
    2. The shortest path distance for the robot from its current location to the box's current location.
    The box-goal distance is weighted by 2 to reflect the additional robot movement likely required for each push.

    This heuristic is non-admissible as it ignores obstacles for the robot's movement and the complex interactions between robot and box movements, and the need for the robot to be on a specific side of the box. It also sums costs for boxes independently.

    # Assumptions:
    - The problem can be represented as a graph where locations are nodes and adjacent facts define edges.
    - The goal state specifies a unique target location for each box.
    - Adjacency is symmetric for robot movement distance calculations (graph edges are bidirectional).
    - The heuristic simplifies robot movement cost and interaction.

    # Heuristic Initialization
    - Collects all unique locations from static facts, initial state, and goals.
    - Parses static facts to build an undirected graph of locations based on `adjacent` predicates for robot movement distance calculations.
    - Precomputes all-pairs shortest path distances between all collected locations using BFS on this graph.
    - Parses goal conditions to map each box to its target goal location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot from the state.
    2. Identify the current location of each box from the state.
    3. Initialize the total heuristic cost to 0.
    4. For each box specified in the goal conditions:
       - Find the box's current location (`loc_b`) and its goal location (`loc_goal`).
       - If the box is already at its goal location (`loc_b == loc_goal`), its contribution to the heuristic is 0. Continue to the next box.
       - If the box is not at its goal location:
         - Calculate the shortest path distance for the box from `loc_b` to `loc_goal` using the precomputed distances on the location graph (`box_distance`). This represents the minimum number of push actions required for the box itself.
         - Calculate the shortest path distance for the robot from its current location (`loc_r`) to the box's current location (`loc_b`) using the precomputed distances (`robot_distance`). This represents the cost for the robot to reach the box.
         - If either `box_distance` or `robot_distance` is infinite (meaning the goal is unreachable for the box, or the box is unreachable by the robot), the state is likely unsolvable, return infinity.
         - The heuristic contribution for this box is `2 * box_distance + robot_distance`. The box distance is weighted by 2 to account for the robot repositioning needed for each push.
       - Add the box's contribution to the total heuristic cost.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Collect all unique locations
        self.all_locations = get_all_locations(static_facts, initial_state, self.goals)

        # 2. Build the undirected graph for robot movement distances
        self.location_graph = build_graph(static_facts)

        # Ensure all collected locations are keys in the graph dictionary, even if isolated
        for loc in self.all_locations:
             if loc not in self.location_graph:
                 self.location_graph[loc] = []

        # 3. Precompute all-pairs shortest path distances
        self.distances = precompute_distances(self.location_graph, self.all_locations)

        # 4. Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break

        # If robot location is not found, state is invalid/unreachable
        if robot_location is None:
             return math.inf

        # Find current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Only consider objects that are boxes and are in the goal
                if obj in self.goal_locations:
                    current_box_locations[obj] = loc

        total_cost = 0

        # Calculate cost for each box
        for box, goal_location in self.goal_locations.items():
            current_location = current_box_locations.get(box)

            # If a box from the goal is not found in the current state, something is wrong
            # or it's an unreachable state.
            if current_location is None:
                 return math.inf # Should not happen in valid states generated by planner

            # If box is already at goal, cost is 0 for this box
            if current_location == goal_location:
                continue

            # Box is not at goal, calculate its contribution

            # Check if locations are in our precomputed distances table
            if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                 # Goal location is unreachable from current box location
                 return math.inf

            box_distance = self.distances[current_location][goal_location]

            # Check if robot location is in our precomputed distances table
            if robot_location not in self.distances or current_location not in self.distances.get(robot_location, {}):
                 # Box location is unreachable by the robot
                 return math.inf

            robot_distance = self.distances[robot_location][current_location]

            # If either distance is infinite, the goal is unreachable for this box/robot combination
            if box_distance == math.inf or robot_distance == math.inf:
                 return math.inf

            # Add contribution for this box: 2 * box_distance + robot_distance
            # This weights the box movement (pushes) more than the robot movement to the box.
            total_cost += 2 * box_distance + robot_distance

        return total_cost
