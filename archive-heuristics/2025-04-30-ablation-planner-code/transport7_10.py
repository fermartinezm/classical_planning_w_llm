# from heuristics.heuristic_base import Heuristic # Assuming this exists in the planner environment

from fnmatch import fnmatch
from collections import deque
import math

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class
class transportHeuristic: # In a real planner, this would likely inherit from Heuristic
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drive, drop)
    required to move each package from its current location to its goal location,
    assuming vehicle availability and sufficient capacity. It sums the estimated
    costs for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location, specified by an `(at package location)` goal fact.
    - Vehicle capacity constraints are simplified; the heuristic assumes a suitable
      vehicle is available when needed for each package's movement segment.
    - Road network is static and bidirectional (implicitly handled by building
      the graph with edges in both directions).
    - The cost of a 'drive' action is 1, regardless of distance (shortest path
      distance is used as the number of drive actions).
    - The cost of 'pick-up' and 'drop' actions is 1.
    - Objects starting with 'v' are vehicles. This is a simplifying assumption based on common naming conventions in benchmarks. A robust heuristic would use PDDL type information.

    # Heuristic Initialization
    - Extracts the road network from static facts to build a graph.
    - Precomputes shortest path distances between all pairs of locations using BFS.
    - Extracts the goal location for each package from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every locatable object (packages and vehicles).
       Packages can be 'at' a location or 'in' a vehicle. Vehicles are always 'at' a location.
       Store vehicle locations in a dictionary. Store package states (either 'at' location or 'in' vehicle name) in another dictionary, focusing only on packages that are part of the goal.
    2. For each package that has a goal location and is not yet at that goal location on the ground:
       a. Determine the package's effective current location. If the package is 'at'
          a location L, its effective location is L. If the package is 'in' a vehicle V,
          its effective location is the location where vehicle V is currently 'at' (retrieved from the vehicle locations dictionary).
       b. Retrieve the package's goal location.
       c. Calculate the shortest path distance (number of drive actions) between the
          effective current location and the goal location using the precomputed distances. If no path exists, the state is likely unsolvable, return infinity.
       d. Estimate the actions needed for this package:
          - If the package is currently 'at' its effective location (not in a vehicle):
            Cost = 1 (pick-up) + shortest_path_distance + 1 (drop).
          - If the package is currently 'in' a vehicle:
            Cost = shortest_path_distance + 1 (drop). (The pick-up is already done).
          - Note: If the effective current location is already the goal location (distance is 0),
            and the package is in a vehicle, the cost is 0 + 1 = 1 (for the drop action).
            If the package was on the ground at the goal, it would have been skipped by the initial check.
       e. Add this estimated cost to the total heuristic value.
    3. The total heuristic value is the sum of the estimated costs for all packages
       that are not at their goal location on the ground.
    4. If all packages are at their goal location on the ground, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and goal conditions.
        Builds the road network graph and precomputes shortest paths.
        """
        # In a real planner, this class would inherit from a Heuristic base class
        # and potentially call super().__init__(task)

        self.goals = task.goals
        self.static = task.static

        # 1. Build the road network graph
        self.graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Roads are bidirectional

        self.locations = list(locations) # Store list of all locations

        # 2. Precompute shortest paths between all pairs of locations using BFS
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

        # 3. Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
            # Ignore other goal types if any, as per domain/example goals.

    def _bfs(self, start_location):
        """
        Performs Breadth-First Search from a start location to find distances
        to all other reachable locations.
        """
        distances = {loc: math.inf for loc in self.locations}
        # Handle case where start_location might not be in self.locations (e.g., malformed problem)
        if start_location not in self.locations:
             # Cannot compute distances from an unknown location
             return distances # All distances remain math.inf

        distances[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()

            # Get neighbors from the graph, handle locations with no roads (shouldn't happen if locations come from road facts)
            neighbors = self.graph.get(current_loc, [])

            for neighbor in neighbors:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state

        # Map packages (from goal_locations) to their current state (location string or vehicle string)
        package_states = {}
        # Map vehicles to their current location string
        vehicle_locations = {}

        # Identify all objects mentioned in the state
        all_objects = set()
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Simple vehicle identification heuristic: objects starting with 'v'
        # This is a potential point of failure if naming conventions differ.
        # A robust heuristic needs PDDL type information from the task object.
        vehicles = {obj for obj in all_objects if obj.startswith('v')}
        # Packages relevant to the heuristic are those in goal_locations.

        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]

             if predicate == "at":
                 obj, loc = parts[1], parts[2]
                 if obj in vehicles:
                     vehicle_locations[obj] = loc
                 # If obj is a package from our goal list and is 'at' a location
                 if obj in self.goal_locations:
                     package_states[obj] = loc

             elif predicate == "in":
                 package, vehicle = parts[1], parts[2]
                 # If a package from our goal list is 'in' a vehicle
                 if package in self.goal_locations:
                     package_states[package] = vehicle # Package is in this vehicle


        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            # This is the goal condition. If met, cost for this package is 0.
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at its goal, cost is 0 for this package

            # Package is not at its goal (on the ground). Calculate cost.
            current_state_obj = package_states.get(package) # This is either a location string or a vehicle string

            if current_state_obj is None:
                 # Package is a goal package but not found in state facts (neither at nor in). Unlikely in valid states.
                 # print(f"Warning: Goal package {package} not found in state facts.")
                 return math.inf # Treat as unreachable

            # Determine if the package is in a vehicle
            # Check if the value stored for the package is a known vehicle name
            is_in_vehicle = current_state_obj in vehicles # Use the identified set of vehicles

            if is_in_vehicle:
                 # Package is in a vehicle
                 vehicle_obj = current_state_obj
                 effective_current_location = vehicle_locations.get(vehicle_obj)
                 if effective_current_location is None:
                     # Package is in a vehicle, but vehicle location is unknown. Invalid state?
                     # print(f"Error: Package {package} is in vehicle {vehicle_obj}, but vehicle location is unknown.")
                     return math.inf # Treat as unreachable
            else:
                 # Package is at a location (on the ground)
                 effective_current_location = current_state_obj
                 # Check if the effective location is a known location from the graph
                 if effective_current_location not in self.locations:
                      # Location mentioned in state is not in the road network. Invalid state?
                      # print(f"Error: Package {package} is at unknown location {effective_current_location}")
                      return math.inf # Treat as unreachable


            # Calculate distance from effective current location to goal location
            # Use get with default math.inf in case goal_location is not in self.locations
            # (e.g., malformed goal or static facts) or if effective_current_location
            # is not a valid key in shortest_paths (shouldn't happen if checked above).
            distance = self.shortest_paths.get(effective_current_location, {}).get(goal_location, math.inf)

            if distance == math.inf:
                # Goal location is unreachable from the package's current effective location.
                return math.inf

            # Calculate cost for this package
            package_cost = 0
            if not is_in_vehicle:
                # Package is on the ground, needs pick-up
                package_cost += 1 # pick-up action

            # Needs transport (drive actions)
            package_cost += distance # number of drive actions

            # Needs drop action
            package_cost += 1 # drop action

            total_cost += package_cost

        return total_cost
