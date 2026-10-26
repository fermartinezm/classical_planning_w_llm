# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Utility functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    from its current location to its goal location. It considers the actions
    required to pick up the package (if on the ground), drive the vehicle
    carrying the package (if needed), and drop the package. The drive cost
    is estimated as the shortest path distance in the road network.

    # Assumptions
    - The goal is to have specific packages at specific ground locations.
    - All relevant locations (initial package locations, goal package locations,
      initial vehicle locations) are part of a connected road network.
    - Vehicle capacity constraints are not explicitly modeled in the heuristic
      value calculation itself, only the fundamental pick/drive/drop steps.
    - Action costs are uniform (cost 1).
    - Vehicle names start with 'v' and package names start with 'p'.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build a graph representation of the road network from static facts.
    - Compute all-pairs shortest paths on the road network graph.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package (either on the ground at L or inside a vehicle V).
    2. Identify the current ground location of every vehicle (at L_v).
    3. Initialize total heuristic cost to 0.
    4. For each package that has a specified goal location:
       a. Determine the package's current state: on the ground at location `current_loc`, or inside vehicle `vehicle` which is at location `vehicle_loc`.
       b. Determine the package's goal location `goal_loc`.
       c. If the package is currently on the ground at `current_loc`:
          - If `current_loc` is not the `goal_loc`:
            - Add 1 to cost (for pick-up).
            - Add the shortest path distance from `current_loc` to `goal_loc` to cost (for driving).
            - Add 1 to cost (for drop).
          - If `current_loc` is the `goal_loc`, cost for this package is 0.
       d. If the package is currently inside `vehicle` which is at `vehicle_loc`:
          - If `vehicle_loc` is not the `goal_loc`:
            - Add the shortest path distance from `vehicle_loc` to `goal_loc` to cost (for driving).
          - Add 1 to cost (for drop).
       e. If any required location is unreachable in the road network, the heuristic returns infinity.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and computing shortest paths.
        """
        # Assuming Heuristic base class handles task assignment
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Assuming goals are always about packages being at locations
                if package.startswith('p'): # Basic check based on naming convention
                    self.goal_locations[package] = location
            # Ignore other potential goal types or malformed goals

        # Build the road network graph and collect all locations.
        self.adj = {}
        all_locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1) # Roads are bidirectional

        self.all_locations = list(all_locations) # Keep a list for consistent iteration

        # Compute all-pairs shortest paths using BFS.
        self.distance = {}
        for start_loc in self.all_locations:
            self.distance[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """
        Performs Breadth-First Search from a start location to find distances
        to all other reachable locations in the road network.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.all_locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            # Check if current_loc exists in adjacency list (it should if it's in all_locations)
            # and has neighbors.
            if current_loc in self.adj:
                for neighbor in self.adj[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles.
        package_location = {} # Maps package -> location (ground or vehicle name)
        vehicle_location = {} # Maps vehicle -> ground location

        # Populate locations from the current state facts.
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3: # Check for predicates with two arguments like (at ?x ?y) or (in ?x ?y)
                predicate, obj1, obj2 = parts
                if predicate == "at":
                    # Assuming obj1 is locatable (vehicle or package) and obj2 is location
                    if obj1.startswith('v'):
                        vehicle_location[obj1] = obj2 # Vehicle obj1 is at ground location obj2
                    elif obj1.startswith('p'):
                         package_location[obj1] = obj2 # Package obj1 is at ground location obj2
                elif predicate == "in":
                    # Assuming (in package vehicle)
                    package, vehicle = obj1, obj2
                    if package.startswith('p') and vehicle.startswith('v'):
                         package_location[package] = vehicle # Package is in vehicle

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that has a goal location.
        for package, goal_location in self.goal_locations.items():
            # If package is not in the state, it's an unexpected situation for a goal package.
            # Returning infinity is safest for unsolvable subproblems.
            if package not in package_location:
                 return float('inf') # Package with goal not found in state

            current_loc_or_vehicle = package_location[package]

            # Check if the package is inside a vehicle.
            if current_loc_or_vehicle.startswith('v'): # Assuming vehicle names start with 'v'
                vehicle = current_loc_or_vehicle
                # Find the vehicle's current ground location
                vehicle_loc = vehicle_location.get(vehicle)

                if vehicle_loc is None:
                    # Vehicle carrying the package is not 'at' any location - unsolvable state?
                    # This shouldn't happen in a valid state representation.
                    return float('inf')

                # Package is in vehicle at vehicle_loc
                if vehicle_loc != goal_location:
                    # Vehicle needs to drive to goal_location
                    # Get distance, return infinity if unreachable
                    drive_cost = self.distance.get(vehicle_loc, {}).get(goal_location, float('inf'))
                    if drive_cost == float('inf'):
                         return float('inf') # Goal location unreachable from vehicle's current location
                    total_cost += drive_cost

                # Package needs to be dropped at goal_location (or vehicle_loc if already there)
                total_cost += 1 # Drop action

            else: # Package is on the ground
                current_loc = current_loc_or_vehicle

                if current_loc != goal_location:
                    # Package needs to be picked up, vehicle drives, package dropped
                    total_cost += 1 # Pick-up action

                    # Need to find a vehicle to pick it up and drive.
                    # Simplification: Assume a vehicle is available at current_loc
                    # and can reach goal_location. The cost is just the drive distance.
                    # We don't model vehicle availability or capacity here.
                    # Get distance, return infinity if unreachable
                    drive_cost = self.distance.get(current_loc, {}).get(goal_location, float('inf'))
                    if drive_cost == float('inf'):
                         return float('inf') # Goal location unreachable from package's current location
                    total_cost += drive_cost

                    total_cost += 1 # Drop action
                # Else: current_loc == goal_location, cost for this package is 0.

        return total_cost
