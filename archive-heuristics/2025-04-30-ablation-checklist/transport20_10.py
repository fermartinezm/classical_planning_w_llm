import sys
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Handle cases where args might be shorter than parts (e.g., matching "(at obj loc)" with "at", "*")
    # This simple zip check works for the patterns used here.
    return True


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It calculates the cost
    for each package independently, ignoring vehicle capacity and potential
    synergies (like one vehicle moving multiple packages).

    # Assumptions
    - Goals for packages are always of the form `(at ?p ?l)`.
    - Roads are bidirectional (if `(road l1 l2)` exists, assume `(road l2 l1)`).
    - Vehicles are always located at some location (`(at ?v ?l)`).
    - Packages are either `(at ?p ?l)` or `(in ?p ?v)`.
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Parse goal conditions to map each package to its goal location.
    - Build a graph of locations based on `road` facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify all vehicle objects.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not at its goal location:

    1.  **Determine Package's Current State:** Is the package `(at ?p ?l_current)` or `(in ?p ?v)`?
    2.  **If `(at ?p ?l_current)` and `l_current != l_goal`:**
        *   The package needs to be picked up (1 action).
        *   A vehicle needs to drive from `l_current` to `l_goal` (distance(l_current, l_goal) actions).
        *   The package needs to be dropped (1 action).
        *   Total cost for this package: 1 + distance(l_current, l_goal) + 1.
    3.  **If `(in ?p ?v)`:**
        *   Find the current location of vehicle `v`, say `l_v`.
        *   If `l_v == l_goal`: The package needs to be dropped (1 action). Total cost: 1.
        *   If `l_v != l_goal`:
            *   The vehicle needs to drive from `l_v` to `l_goal` (distance(l_v, l_goal) actions).
            *   The package needs to be dropped (1 action).
            *   Total cost for this package: distance(l_v, l_goal) + 1.
    4.  **Sum Costs:** The total heuristic value is the sum of the costs calculated for each package not at its goal. If any required location is unreachable, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        location graph, computing distances, and identifying vehicles.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all objects/locations

        # 1. Parse goals: package -> goal_location
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.package_goals[package] = location
            # Assuming only 'at' goals for packages

        # 2. Build location graph from 'road' facts and collect all locations
        self.location_graph = {}
        self.all_locations = set()

        # Collect locations from road facts
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Assuming bidirectional roads
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Collect locations from initial state 'at' facts
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.all_locations.add(loc)

        # Collect locations from goal 'at' facts
        for loc in self.package_goals.values():
             self.all_locations.add(loc)

        # Ensure all collected locations are keys in the graph dictionary
        for loc in self.all_locations:
             self.location_graph.setdefault(loc, [])

        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify vehicles (assuming vehicles are objects with capacity)
        self.vehicles = set()
        for fact in self.initial_state:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, _ = get_parts(fact)
                self.vehicles.add(vehicle)
        # Also collect vehicles that might be mentioned in 'in' predicates in goals/initial state
        for fact in self.initial_state:
             if match(fact, "in", "*", "*"):
                  _, package, vehicle = get_parts(fact)
                  self.vehicles.add(vehicle)
        for goal in self.goals:
             if match(goal, "in", "*", "*"):
                  _, package, vehicle = get_parts(goal)
                  self.vehicles.add(vehicle)


    def _bfs(self, start_loc):
        """Performs BFS from start_loc to find distances to all other locations."""
        distances = {loc: float('inf') for loc in self.all_locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Check if current_loc has any roads defined
            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
         """Retrieves precomputed distance, returns infinity for unknown/unreachable."""
         if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This should ideally not happen if all relevant locations are collected
             # but serves as a safeguard.
             return float('inf')
         return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current locations/containment for all objects
        current_locations = {} # obj -> location
        package_in_vehicle = {} # package -> vehicle
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_locations[obj] = loc
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                _, package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # Find package's current state: location or vehicle it's in
            pkg_current_loc = current_locations.get(package)
            pkg_current_vehicle = package_in_vehicle.get(package)

            # Case 1: Package is already at the goal location
            if pkg_current_loc == goal_location:
                continue # Cost is 0 for this package

            # Case 2: Package is on the ground at l_current (not goal)
            if pkg_current_loc is not None: # It's 'at' a location
                l_current = pkg_current_loc
                # Cost: Pick up (1) + Drive (distance) + Drop (1)
                dist = self.get_distance(l_current, goal_location)
                if dist == float('inf'):
                     return float('inf') # Goal unreachable for this package
                total_cost += 1 + dist + 1

            # Case 3: Package is inside a vehicle v
            elif pkg_current_vehicle is not None: # It's 'in' a vehicle
                v = pkg_current_vehicle
                # Find vehicle's location
                l_v = vehicle_locations.get(v)
                if l_v is None:
                     # Vehicle is not 'at' any location? Invalid state or vehicle not in state?
                     # Assume vehicles are always 'at' a location if package is 'in' them.
                     # If vehicle location is unknown, this path is impossible.
                     return float('inf') # Cannot determine vehicle location

                # Cost: Drive vehicle from l_v to goal_location (distance) + Drop (1)
                dist = self.get_distance(l_v, goal_location)
                if dist == float('inf'):
                     return float('inf') # Goal unreachable for this package
                total_cost += dist + 1

            # Else: Package is neither 'at' nor 'in'? Invalid state?
            # Based on domain, packages are always 'at' or 'in'. If not found, something is wrong.
            # For robustness, we could return infinity or a large number.
            # Given the problem structure, this case likely indicates an issue with state representation
            # or an unreachable goal if the package isn't in the state facts at all.
            # We'll implicitly skip if package isn't in current_locations or package_in_vehicle.
            # If a package has a goal but isn't in the state at all, it's unreachable.
            elif package in self.package_goals: # Package has a goal but no status in state
                 return float('inf')


        return total_cost

# Note: This code assumes the existence of a base class `Heuristic`
# in a module `heuristics.heuristic_base` with the signature:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass
# And a `node` object with a `state` attribute (frozenset of fact strings).
# And a `task` object with `goals`, `initial_state`, and `static` attributes
# (all frozensets of fact strings).
