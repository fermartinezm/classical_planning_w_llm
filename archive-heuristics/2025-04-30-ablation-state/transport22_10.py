# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic # Uncomment this line in the actual file

from fnmatch import fnmatch
from collections import deque
import math # Import math for math.inf

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing whitespace
    fact_str = str(fact).strip()
    # Check if it looks like a PDDL fact (starts with '(' and ends with ')')
    if fact_str.startswith('(') and fact_str.endswith(')'):
        return fact_str[1:-1].split()
    # Handle cases that might not be standard facts, though state facts should be
    # Return empty list or raise error for malformed facts? Let's return empty list.
    return []


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assume Heuristic base class is imported elsewhere, e.g., from heuristics.heuristic_base
# class Heuristic: pass # Dummy definition if not imported

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the current location of each package
    (on the ground or in a vehicle) and the shortest path distance in the road
    network to its goal location. It sums the estimated costs for each package
    independently, ignoring vehicle capacity and availability constraints.

    # Assumptions
    - The road network defined by `(road l1 l2)` facts is static and bidirectional.
    - The cost of a `drive` action is 1 for each step in the shortest path.
    - `pick-up` and `drop` actions cost 1.
    - Vehicle capacity and availability are ignored (relaxed problem).
    - Any package can be transported by any vehicle.
    - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles (based on example naming conventions).
    - All packages mentioned in the goal exist and are either 'at' a location or 'in' a vehicle in any valid state.

    # Heuristic Initialization
    - Extracts all location objects involved in `road` predicates from the static facts.
    - Builds an adjacency list representation of the road network graph based on `road` predicates. Assumes roads are bidirectional.
    - Computes all-pairs shortest paths between locations using Breadth-First Search (BFS). Stores distances in `self.shortest_paths[start_loc][end_loc]`.
    - Extracts the goal location for each package from the task's goal conditions (`(at package location)` facts).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If `task.goals` is a subset of the state facts, return 0.
    2. Build lookup dictionaries for the current state:
       - `current_package_locations`: Maps package name to its location if `(at package location)` is true.
       - `packages_in_vehicle`: Maps package name to the vehicle name if `(in package vehicle)` is true.
       - `current_vehicle_locations`: Maps vehicle name to its location if `(at vehicle location)` is true.
       These are populated by iterating through the state facts. Assumes packages start with 'p' and vehicles with 'v'.
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` that has a goal location `goal_l` (extracted during initialization):
       a. Check if the package `p` is currently on the ground at its goal location (`package` in `current_package_locations` and `current_package_locations[package] == goal_l`). If it is, the cost for this package is 0, and we move to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current status based on the lookup dictionaries:
             - If `package` is in `current_package_locations`: The package is on the ground at `current_location = current_package_locations[package]`.
             - If `package` is in `packages_in_vehicle`: The package is inside `vehicle_name = packages_in_vehicle[package]`. Find the vehicle's location `vehicle_location = current_vehicle_locations.get(vehicle_name)`.
             - If the package is neither 'at' a location nor 'in' a vehicle (or the vehicle's location is unknown), the state is inconsistent or unsolvable from this point; return `math.inf`.
          ii. Calculate the estimated cost for this package:
              - If on the ground at `current_location` (`current_location != goal_l`):
                  - Estimated cost = 1 (pick-up) + `dist(current_location, goal_l)` (drive) + 1 (drop).
              - If inside a vehicle at `vehicle_location`:
                  - Estimated cost = `dist(vehicle_location, goal_l)` (drive) + 1 (drop).
              - `dist(l1, l2)` is the shortest path distance between `l1` and `l2` computed during initialization. If the goal location is unreachable (`dist` is `math.inf`), return `math.inf` for the total heuristic.
          iii. Add the estimated cost for the current package to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions (frozenset of facts).
        static_facts = task.static  # Static facts (frozenset of facts).

        # Extract all locations and build the road graph
        self.locations = set()
        self.road_graph = {} # Adjacency list: location -> set of connected locations

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional unless specified otherwise
                self.road_graph.setdefault(l2, set()).add(l1)

        # Ensure all locations found are in the graph keys, even if they have no roads
        for loc in self.locations:
             self.road_graph.setdefault(loc, set())

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Add other goal types if necessary, but 'at' is the primary one for packages

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find shortest
        distances to all other nodes in the road graph.
        Returns a dictionary mapping location -> distance.
        """
        distances = {loc: math.inf for loc in self.locations}
        # Handle case where start_node might not be in self.locations (e.g., malformed data)
        if start_node not in self.locations:
             # Cannot compute distances from an unknown location
             return distances # All distances remain infinity

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc exists in the graph keys before iterating
            # This check is redundant if self.road_graph was initialized correctly from self.locations
            # but doesn't hurt.
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Build lookup dictionaries for current locations/containment
        current_package_locations = {} # package_name -> location_name (if at)
        packages_in_vehicle = {} # package_name -> vehicle_name (if in)
        current_vehicle_locations = {} # vehicle_name -> location_name (if at)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj_name = parts[1]
                loc_name = parts[2]
                # Simple check for type based on naming convention from examples
                # A more robust approach would parse domain types.
                if obj_name.startswith('p'): # Assuming objects starting with 'p' are packages
                     current_package_locations[obj_name] = loc_name
                elif obj_name.startswith('v'): # Assuming objects starting with 'v' are vehicles
                     current_vehicle_locations[obj_name] = loc_name
                # Add other types if needed, but 'at' only applies to locatables (vehicles, packages)
            elif predicate == "in" and len(parts) == 3:
                package_name = parts[1]
                vehicle_name = parts[2]
                # Assuming packages start with 'p' and vehicles start with 'v'
                if package_name.startswith('p') and vehicle_name.startswith('v'):
                     packages_in_vehicle[package_name] = vehicle_name
                # Add other types if needed, but 'in' only applies to package in vehicle


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal
            if package in current_package_locations and current_package_locations[package] == goal_location:
                continue # Package is already at its goal, cost is 0 for this package

            # Package is not at goal, calculate cost
            if package in current_package_locations:
                # Package is on the ground at current_package_locations[package]
                current_location = current_package_locations[package]
                # Cost: pick-up (1) + drive (dist) + drop (1)
                # Need to handle cases where current_location or goal_location might not be in self.locations
                # (e.g., if the problem file has locations not connected by roads, though unlikely for goal/initial states)
                if current_location not in self.locations or goal_location not in self.locations:
                     # This indicates a problem with the input data or domain definition
                     return math.inf # Treat as unsolvable

                dist = self.shortest_paths.get(current_location, {}).get(goal_location, math.inf)
                if dist == math.inf:
                    # Goal is unreachable from current location
                    return math.inf # State is likely unsolvable
                total_cost += 1 + dist + 1 # pick + drive + drop

            elif package in packages_in_vehicle:
                # Package is inside a vehicle
                vehicle_name = packages_in_vehicle[package]
                vehicle_location = current_vehicle_locations.get(vehicle_name)

                if vehicle_location is None:
                     # Vehicle containing the package has no location listed in state.
                     # This indicates an inconsistent state or parsing issue.
                     # Treat as unsolvable from here.
                     return math.inf

                # Cost: drive (dist) + drop (1)
                # Need to handle cases where vehicle_location or goal_location might not be in self.locations
                if vehicle_location not in self.locations or goal_location not in self.locations:
                     # This indicates a problem with the input data or domain definition
                     return math.inf # Treat as unsolvable

                dist = self.shortest_paths.get(vehicle_location, {}).get(goal_location, math.inf)
                if dist == math.inf:
                     # Goal is unreachable from vehicle location
                     return math.inf # State is likely unsolvable
                total_cost += dist + 1 # drive + drop
            else:
                 # Package is not 'at' any location and not 'in' any vehicle.
                 # This indicates an inconsistent state. Treat as unsolvable.
                 return math.inf


        return total_cost
