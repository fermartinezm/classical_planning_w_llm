# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if needed

# Define a dummy Heuristic base class for standalone testing if necessary
# In a real system, this would be provided by the planning framework.
class Heuristic:
    def __init__(self, task):
        self.task = task

    def __call__(self, node):
        raise NotImplementedError("Heuristic calculation must be implemented by subclasses.")

import collections
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all
    misplaced packages to their goal locations. It sums the estimated
    cost for each package independently, ignoring vehicle capacity and
    potential synergies from carrying multiple packages in one trip.
    The estimated cost for a single package includes:
    - 1 action to pick up (if on the ground).
    - The shortest path distance (number of drive actions) for a vehicle
      to travel from the package's current location (or its vehicle's
      location) to the package's goal location.
    - 1 action to drop off.

    # Assumptions
    - The road network is static and defines movement costs (1 per drive).
    - Any location is reachable from any other location if a path exists.
    - Vehicle capacity constraints are ignored.
    - A vehicle is always available to pick up a package if needed at its location.
    - Packages can be moved independently.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment status for all packages
       and vehicles.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Check if the package is already AT its goal location on the ground.
          If yes, this package contributes 0 to the heuristic.
       b. If the package is not at its goal:
          i. Determine the package's effective "origin" location:
             - If the package is on the ground at location L, the origin is L.
             - If the package is inside a vehicle V, find the location of V;
               this location is the origin.
          ii. Determine the package's goal location G.
          iii. Calculate the shortest path distance (number of drive actions)
               between the origin location and the goal location using the
               precomputed distances. Let this be D. If the goal is unreachable
               from the origin, the cost for this package is infinite.
          iv. Estimate the actions needed for this package:
              - If the package was on the ground at the origin: 1 (pick-up) + D (drive) + 1 (drop).
              - If the package was inside a vehicle at the origin: D (drive) + 1 (drop).
              If D is infinite, the total cost for the package is infinite.
          v. Add this estimated cost to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph for distance calculations.
        """
        super().__init__(task) # Call base class constructor

        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                # Goal is (at package location)
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = collections.defaultdict(set)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_loc, all_locations):
        """
        Performs BFS from a start location to find distances to all other locations.
        Returns a dictionary mapping location to distance.
        """
        distances_from_start = {loc: float('inf') for loc in all_locations}
        distances_from_start[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances_from_start[current_loc]

            # Check if current_loc is in the graph keys before iterating neighbors
            # This handles cases where a location might be in all_locations but have no roads
            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances_from_start[neighbor] == float('inf'):
                        distances_from_start[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances_from_start

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatables (packages, vehicles) to their current status.
        # For packages: location string (if on ground) or vehicle name string (if in vehicle).
        # For vehicles: location string.
        current_status = {}
        # Map vehicles to their locations.
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                # (at obj loc)
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                # Assuming vehicles start with 'v' based on examples
                # A more robust check would involve parsing object types from the problem file
                # or checking if the object appears in vehicle-specific predicates like 'capacity'.
                # For this domain-dependent heuristic, the 'v' prefix is likely sufficient.
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                # (in package vehicle)
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location specified.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already AT its goal location on the ground.
            # The goal predicate is (at package location).
            if f"(at {package} {goal_location})" in state:
                continue # This package is already at its goal, contributes 0 cost.

            # Package is not at its goal location. Calculate its cost.
            package_cost = 0
            l_origin = None # The location where the package starts its journey from

            if package not in current_status:
                 # Package status is unknown - cannot estimate cost.
                 # This indicates an invalid state representation or a package
                 # that doesn't exist according to the state. Treat as infinite cost.
                 package_cost = float('inf')
            else:
                status = current_status[package]

                if status.startswith('v'):
                    # Package is in a vehicle V
                    vehicle = status
                    if vehicle in vehicle_locations:
                        l_origin = vehicle_locations[vehicle]
                        # Cost includes drive from vehicle's location to goal + drop
                        package_cost += 1 # Cost for drop action
                    else:
                         # Vehicle location unknown - invalid state?
                         # A package is in a vehicle, but the vehicle's location is not known.
                         package_cost = float('inf')
                else:
                    # Package is on the ground at status (a location)
                    l_origin = status
                    # Cost includes pick-up + drive from current location to goal + drop
                    package_cost += 1 # Cost for pick-up action
                    package_cost += 1 # Cost for drop action

            # Add drive cost if origin and goal are known and reachable
            if package_cost != float('inf') and l_origin is not None and goal_location is not None:
                 if l_origin in self.distances and goal_location in self.distances[l_origin]:
                     drive_distance = self.distances[l_origin][goal_location]
                     if drive_distance == float('inf'):
                         # Goal is unreachable from origin
                         package_cost = float('inf') # Unreachable goal means infinite cost
                     else:
                         package_cost += drive_distance
                 else:
                     # Origin or goal location not found in precomputed distances
                     # This indicates a malformed problem/state where a location exists
                     # in state/goals but not in the static road network.
                     package_cost = float('inf') # Cannot estimate distance

            total_cost += package_cost

        # If total_cost is infinity, return infinity. Otherwise, return the integer cost.
        # Heuristic should be 0 only at goal. If total_cost is 0, it means all packages
        # were found to be at their goal locations (checked by `f"(at {package} {goal_location})" in state`).
        # This matches the requirement that h=0 only for goal states.
        return int(total_cost) if total_cost != float('inf') else float('inf')
