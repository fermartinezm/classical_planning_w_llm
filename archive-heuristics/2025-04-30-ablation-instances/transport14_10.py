# Imports needed
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions (copied from Logistics example, they are general)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(start_node, graph):
    """Computes shortest path distances from start_node to all reachable nodes in the graph."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        for neighbor in graph.get(current_node, set()):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport each package
    to its goal location independently. It considers the cost of picking up,
    dropping off, and the shortest path distance a vehicle would need to travel.

    # Assumptions
    - Actions (pick-up, drop, drive) have a cost of 1.
    - Capacity constraints of vehicles are ignored.
    - Multiple packages can be transported simultaneously (implicitly, as costs are summed independently).
    - The shortest path distance between locations on the road network is a reasonable estimate for vehicle travel cost.
    - Roads are bidirectional (if road A-B exists, road B-A also exists).

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representing the road network from static facts.
    - Collects all relevant locations from static facts, initial state, and goals.
    - Computes all-pairs shortest path distances between all collected locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment status for every package and vehicle by parsing the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Check if the package is currently on the ground at its goal location. If yes, add 0 cost for this package and move to the next.
       b. If the package is not at its goal location, determine its current status: Is it on the ground at some `current_l`, or is it inside a vehicle `v`?
       c. If the package is on the ground at `current_l` (`current_l` is not the goal):
          - Estimate the cost for this package as: 1 (pick-up) + shortest_distance(`current_l`, `goal_l`) + 1 (drop).
          - If `goal_l` is unreachable from `current_l` via roads, the state is likely unsolvable; return infinity.
       d. If the package is inside a vehicle `v`, find the vehicle's current location `vehicle_l`.
          - Estimate the cost for this package as: shortest_distance(`vehicle_l`, `goal_l`) + 1 (drop).
          - If `goal_l` is unreachable from `vehicle_l` via roads, the state is likely unsolvable; return infinity.
       e. Add the estimated cost for this package to the total heuristic cost.
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the road
        graph, and precomputing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        all_locations = set()

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
                all_locations.add(location)

        # Build the road graph and collect all locations mentioned in roads
        self.road_graph = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional
                all_locations.add(l1)
                all_locations.add(l2)

        # Collect locations from initial state 'at' facts
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                all_locations.add(loc)

        # Compute all-pairs shortest paths
        self.distances = {}
        for loc in all_locations:
             self.distances[loc] = bfs(loc, self.road_graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        at_locations = {}
        in_vehicles = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                at_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                _, pkg, veh = get_parts(fact)
                in_vehicles[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # The goal is (at package location), not (in package vehicle) at location
            if package in at_locations and at_locations[package] == goal_location:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal (either wrong location or inside a vehicle)
            if package in at_locations:
                # Package is on the ground at current_location, which is not the goal
                current_location = at_locations[package]
                
                # Estimate travel cost: pick-up + drive + drop
                # Need distance from current_location to goal_location
                dist_info = self.distances.get(current_location)
                if dist_info is None:
                     # Current location is not in the graph (shouldn't happen if locations collected correctly)
                     # This indicates an issue with graph building or state representation
                     return float('inf')
                dist = dist_info.get(goal_location)

                if dist is None:
                    # Goal location is unreachable from current location
                    return float('inf')

                # Cost = pick-up (1) + drive (dist) + drop (1)
                total_cost += 1 + dist + 1

            elif package in in_vehicles:
                # Package is inside a vehicle
                vehicle = in_vehicles[package]
                
                # Find the vehicle's location
                vehicle_location = at_locations.get(vehicle)
                if vehicle_location is None:
                    # Vehicle location is unknown (invalid state?)
                    # Every locatable (including vehicles) should have an 'at' fact
                    return float('inf')

                # Estimate travel cost: drive + drop
                # Need distance from vehicle_location to goal_location
                dist_info = self.distances.get(vehicle_location)
                if dist_info is None:
                     # Vehicle location is not in the graph
                     return float('inf')
                dist = dist_info.get(goal_location)

                if dist is None:
                    # Goal location is unreachable from vehicle location
                    return float('inf')

                # Cost = drive (dist) + drop (1)
                total_cost += dist + 1
            
            # else: package is not in at_locations and not in_vehicles. This indicates
            # an invalid state where a package is not located anywhere. Return infinity.
            else:
                 return float('inf')


        return total_cost
