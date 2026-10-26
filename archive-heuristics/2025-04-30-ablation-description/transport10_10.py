from fnmatch import fnmatch
from collections import deque
import math

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure input is treated as a string
    fact_str = str(fact).strip()
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Handle facts that might not be in the standard (predicate arg1 arg2) format
    return fact_str.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package to its goal location, independently of other
    packages and vehicle capacity constraints. It sums the estimated costs for
    all packages that are not yet at their final destination on the ground.

    # Assumptions
    - Goals are always of the form (at ?package ?location).
    - Any vehicle can transport any package (capacity is ignored).
    - Vehicles are always available when needed at a package's location.
    - The cost of driving between two locations is the shortest path distance
      in the road network graph. Each drive action costs 1. Pick-up and drop
      actions also cost 1.
    - The road network is static and defined by (road ?l1 ?l2) facts.
    - Roads are assumed to be bidirectional if a (road l1 l2) fact exists.
    - Object types (package, vehicle, location, size) are inferred from the
      predicate positions they occupy in the initial state, goals, and static facts.

    # Heuristic Initialization
    - Identify all packages, vehicles, locations, and sizes from the task definition
      by inspecting the predicates they appear in across the initial state, goals,
      and static facts.
    - Store the goal location for each package specified in the goals.
    - Build the road network graph from static (road ?l1 ?l2) facts.
    - Precompute all-pairs shortest path distances between all identified locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:
    1. Initialize the total heuristic cost to 0.
    2. Determine the current location or containment status for all packages
       and the current location for all vehicles by inspecting the current state facts.
       Store these in dictionaries (package -> location/vehicle, vehicle -> location).
    3. For each package that has a specified goal location:
       a. Find the package's current status (location or vehicle).
       b. If the package's current status is its goal location (meaning it's on the ground
          at the goal location), this package contributes 0 to the heuristic. Continue to
          the next package.
       c. If the package is on the ground at a location different from its goal:
          - Let `current_loc` be the package's current location.
          - Estimate the cost for this package as 1 (pick-up) + shortest_distance(`current_loc`, `goal_location`) + 1 (drop).
          - Add this cost to the total heuristic.
       d. If the package is inside a vehicle:
          - Find the current location of the vehicle. Let this be `current_vehicle_loc`.
          - If `current_vehicle_loc` is the package's goal location:
            - Estimate the cost as 1 (drop).
            - Add this cost to the total heuristic.
          - If `current_vehicle_loc` is different from the package's goal location:
            - Estimate the cost as shortest_distance(`current_vehicle_loc`, `goal_location`) + 1 (drop).
            - Add this cost to the total heuristic.
    4. The total heuristic value is the sum of the estimated costs for all packages.
       If any required shortest distance was infinite (meaning the goal location is
       unreachable from the package's current effective location in the road network),
       the total heuristic will be a large number (1e9) representing infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest path distances.
        """
        super().__init__(task) # Call the base class constructor

        # 1. Identify object types (packages, vehicles, locations, sizes) based on predicate arguments
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()

        all_facts = set(self.task.initial_state) | set(self.goals) | set(self.static)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts or len(parts) < 2: continue # Skip malformed or trivial facts

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                # (at ?locatable ?location)
                # parts[1] is locatable (package or vehicle)
                self.locations.add(parts[2])
            elif predicate == 'in' and len(parts) == 3:
                 # (in ?package ?vehicle)
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])
            elif predicate == 'capacity' and len(parts) == 3:
                 # (capacity ?vehicle ?size)
                 self.vehicles.add(parts[1])
                 self.sizes.add(parts[2])
            elif predicate == 'road' and len(parts) == 3:
                 # (road ?location ?location)
                 self.locations.add(parts[1])
                 self.locations.add(parts[2])
            elif predicate == 'capacity-predecessor' and len(parts) == 3:
                 # (capacity-predecessor ?size ?size)
                 self.sizes.add(parts[1])
                 self.sizes.add(parts[2])

        # 2. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at ?package ?location)
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    # Only store goals for objects identified as packages
                    if package in self.packages:
                         self.goal_locations[package] = location
                    # Note: If a package only appears in 'at' facts and never 'in',
                    # it won't be in self.packages by this inference method.
                    # This heuristic focuses on packages that can be transported ('in' predicate).


        # 3. Build the road network graph
        self.road_graph = {loc: [] for loc in self.locations}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    loc1, loc2 = parts[1], parts[2]
                    # Add directed edge
                    self.road_graph[loc1].append(loc2)
                    # Add reverse edge if roads are bidirectional (common in transport)
                    # The example instance files show bidirectional roads.
                    # If the domain intended unidirectional, this line should be removed.
                    # Assuming bidirectional based on examples.
                    self.road_graph[loc2].append(loc1)

        # Remove duplicates from adjacency lists (in case roads are listed multiple times or bidirectionally added)
        for loc in self.road_graph:
             self.road_graph[loc] = list(set(self.road_graph[loc]))

        # Ensure all locations identified are in the graph structure, even if they have no roads
        for loc in self.locations:
            if loc not in self.road_graph:
                self.road_graph[loc] = []


        # 4. Precompute all-pairs shortest path distances using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = {loc: math.inf for loc in self.locations}
            self.distances[start_loc][start_loc] = 0
            queue = deque([start_loc])

            while queue:
                u = queue.popleft()
                # Ensure u is a valid key in the graph (should be if populated from self.locations)
                if u in self.road_graph:
                    for v in self.road_graph[u]:
                        # Ensure v is a valid location key in distances (should be if populated from self.locations)
                        if v in self.distances[start_loc] and self.distances[start_loc][v] == math.inf:
                            self.distances[start_loc][v] = self.distances[start_loc][u] + 1
                            queue.append(v)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts)

        # Determine current locations/containment for packages and vehicles
        package_current_state = {} # Maps package to its location (string) or vehicle (string)
        vehicle_current_location = {} # Maps vehicle to its location (string)

        for fact in state:
            parts = get_parts(fact)
            if not parts or len(parts) < 2: continue

            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                    vehicle_current_location[obj] = loc
            elif predicate == 'in' and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                if pkg in self.packages and veh in self.vehicles:
                    package_current_state[pkg] = veh

        total_cost = 0
        unreachable_goal_found = False # Flag to indicate if any package goal is unreachable

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not found in the current state facts, it's an unexpected state.
            # In a valid search, all objects should have a defined location/containment.
            # If it's missing, it implies an issue with the state representation or domain.
            # For robustness, we assign a large penalty.
            if package not in package_current_state:
                 # print(f"Warning: Package {package} with goal {goal_location} not found in current state.")
                 unreachable_goal_found = True
                 break # No need to check other packages if one is unreachable

            current_state_of_package = package_current_state[package]

            # Check if the package is already at its goal location on the ground
            # The goal is (at package goal_location). If package_current_state[package] is goal_location,
            # it means the fact (at package goal_location) is present in the state.
            if current_state_of_package == goal_location:
                 continue # Cost is 0 for this package

            # Package is not at its goal location on the ground. Estimate cost.

            # Case 1: Package is on the ground at a different location
            if current_state_of_package in self.locations: # Check if the value is a location string
                current_package_location = current_state_of_package
                # Cost = pick-up + drive + drop
                dist = self.distances.get(current_package_location, {}).get(goal_location, math.inf)

                if dist == math.inf:
                    # Goal location is unreachable from the package's current location
                    unreachable_goal_found = True
                    break # No need to check other packages
                else:
                    # Cost = 1 (pick) + dist (drive) + 1 (drop)
                    total_cost += 1 + dist + 1

            # Case 2: Package is inside a vehicle
            elif current_state_of_package in self.vehicles: # Check if the value is a vehicle string
                vehicle = current_state_of_package
                # Find the vehicle's current location
                current_vehicle_location = vehicle_current_location.get(vehicle)

                if current_vehicle_location is None:
                    # Vehicle location not found in state? Should not happen in valid states.
                    # Treat as unreachable.
                    # print(f"Warning: Vehicle {vehicle} containing package {package} has no location in state.")
                    unreachable_goal_found = True
                    break # No need to check other packages

                # If vehicle is at the goal location, only drop is needed
                elif current_vehicle_location == goal_location:
                    total_cost += 1 # Cost = drop

                # If vehicle is not at the goal location, need to drive and drop
                else:
                    # Cost = drive + drop
                    dist = self.distances.get(current_vehicle_location, {}).get(goal_location, math.inf)
                    if dist == math.inf:
                         # Goal location is unreachable from the vehicle's current location
                         unreachable_goal_found = True
                         break # No need to check other packages
                    else:
                         total_cost += dist + 1
            else:
                 # The package's current_state is neither a location nor a vehicle? Unexpected.
                 # print(f"Warning: Package {package} in unexpected state: {current_state_of_package}")
                 unreachable_goal_found = True
                 break # Indicate unsolvable


        # Return a large number if any goal was found to be unreachable
        if unreachable_goal_found:
            return 1e9 # A large finite number representing infinity for greedy search

        # Return the estimated cost.
        return total_cost
