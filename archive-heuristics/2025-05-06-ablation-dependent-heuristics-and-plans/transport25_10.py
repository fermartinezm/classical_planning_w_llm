# from heuristics.heuristic_base import Heuristic # Assuming this base class exists and is imported

from fnmatch import fnmatch
from collections import deque

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Handle cases that are not standard PDDL facts (e.g., comments, malformed strings)
        # Returning empty list is safer than erroring
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is sufficient to match the pattern
    if len(parts) < len(args):
        return False
    # Use zip to compare parts and args up to the length of the shorter sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

# Define the heuristic class
class transportHeuristic(Heuristic): # Assuming Heuristic is imported
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package independently.
    The cost for a package is estimated based on its current location (or the vehicle's
    location if it's inside a vehicle) and its goal location, considering the shortest
    path distance in the road network.

    # Assumptions
    - The goal is to move specific packages to specific locations.
    - Roads are bidirectional (based on example instance files).
    - Vehicle capacity and availability are not explicitly modeled in the cost calculation,
      except for the actions required (pick-up, drop, drive). We assume a suitable
      vehicle is available when needed.
    - The cost of each action (pick-up, drop, drive) is 1. The cost of driving
      is the number of road segments traversed (shortest path distance).
    - Vehicle names start with 'v'.

    # Heuristic Initialization
    - Extracts the goal location for each package that appears in an 'at' goal predicate.
    - Builds a graph of locations connected by roads from the static facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that has a goal location specified and is not yet at that goal:
    1. Find the current status of the package:
       - Is it at a location `current_l`? Check for `(at p current_l)` in the state.
       - Is it inside a vehicle `v`? Check for `(in p v)` in the state.
    2. Determine the effective current location for distance calculation:
       - If `(at p current_l)`, the effective location is `current_l`.
       - If `(in p v)`, find the vehicle's location `(at v vehicle_l)` in the state. The effective location is `vehicle_l`.
       - If the package's status is unknown or the vehicle's location is unknown, assign a large penalty.
    3. Calculate the estimated cost for package `p`:
       - If `(at p current_l)` and `current_l != goal_l`:
         - Needs `pick-up` (1 action).
         - Needs `drive` from `current_l` to `goal_l`. Cost is `dist(current_l, goal_l)` actions.
         - Needs `drop` (1 action).
         - Total cost for package `p`: `1 + dist(current_l, goal_l) + 1 = 2 + dist(current_l, goal_l)`.
       - If `(in p v)` and `(at v vehicle_l)` and `vehicle_l != goal_l`:
         - Needs `drive` from `vehicle_l` to `goal_l`. Cost is `dist(vehicle_l, goal_l)` actions.
         - Needs `drop` (1 action).
         - Total cost for package `p`: `dist(vehicle_l, goal_l) + 1`.
       - If `(in p v)` and `(at v vehicle_l)` and `vehicle_l == goal_l`:
         - Needs `drop` (1 action).
         - Total cost for package `p`: `1`.
       - If the goal location is unreachable from the effective current location, assign a large penalty.
    4. Sum the estimated costs for all packages that are not at their goal location.
    5. If all packages are at their goal location (checked by `self.goals <= state`), the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the road network graph to compute shortest paths.
        """
        # Assuming the base class constructor handles task assignment or it's not needed
        # super().__init__(task)

        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package that appears in the goal 'at' predicate.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        self.locations = list(locations) # List of all locations for BFS

        # Compute shortest path distances between all pairs of locations using BFS.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

        # Define a large penalty for unreachable or unknown states
        self.UNREACHABLE_PENALTY = 1000 # Use a large finite number

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all other nodes.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in distances:
             # Start node is not in the graph built from road facts.
             # No paths exist from it to any location in the graph.
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            if current_loc in self.road_graph: # Check if current_loc has neighbors
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        # Lookup the precomputed distance. Returns inf if loc1 or loc2 is not in the graph
        # or if loc2 is unreachable from loc1.
        return self.shortest_paths.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the goal is reached, the heuristic must be 0.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located.
        # Maps object (package or vehicle) to its location or vehicle it's in
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj, location = args
                current_locations[obj] = location
            elif predicate == "in" and len(args) == 2:
                package, vehicle = args
                current_locations[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location using the 'at' predicate.
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost for this package is 0

            # Package is not at its goal. Calculate cost.
            current_status = current_locations.get(package)

            if current_status is None:
                 # Package location/status is unknown in the current state.
                 # This indicates an unexpected state representation or missing fact.
                 total_cost += self.UNREACHABLE_PENALTY
                 continue

            effective_current_location = None
            cost_this_package_actions = 0 # Actions other than driving (pick/drop)

            # Case 1: Package is at a location (not the goal).
            if current_status in self.locations: # Check if the status is a known location string
                effective_current_location = current_status
                # Needs pick-up (1), drive (dist), drop (1)
                cost_this_package_actions = 2 # for pick-up and drop

            # Case 2: Package is inside a vehicle.
            # Check if current_status is a key in current_locations (meaning we know its location)
            # and if it starts with 'v' (assuming vehicle naming convention).
            elif current_status in current_locations and current_status.startswith("v"):
                vehicle_name = current_status
                # Find the vehicle's current location.
                vehicle_location = current_locations.get(vehicle_name)

                if vehicle_location is None or vehicle_location not in self.locations:
                     # Vehicle location unknown or invalid.
                     total_cost += self.UNREACHABLE_PENALTY
                     continue

                effective_current_location = vehicle_location
                # Needs drive (dist), drop (1)
                cost_this_package_actions = 1 # for drop

            else:
                 # Unexpected package status (e.g., 'in' something not a vehicle, or 'at' something not a location)
                 total_cost += self.UNREACHABLE_PENALTY
                 continue # Move to the next package

            # Now calculate the driving cost based on effective_current_location
            if effective_current_location:
                dist = self.get_distance(effective_current_location, goal_location)

                if dist == float('inf'):
                    # Goal location is unreachable from the effective current location.
                    total_cost += self.UNREACHABLE_PENALTY
                else:
                    total_cost += cost_this_package_actions + dist # Add drive cost

            else:
                 # Should be covered by the checks above, but as a fallback
                 total_cost += self.UNREACHABLE_PENALTY


        return total_cost
