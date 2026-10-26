# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import deque

# Define a dummy Heuristic base class if not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """
    Performs BFS to find shortest distances from start_node to all other nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph, no paths possible
         return distances

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is still valid in case graph changed unexpectedly (it shouldn't)
        if current_node in graph:
            for neighbor in graph.get(current_node, []): # Use .get for safety
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, considering pick-up, drop, and vehicle
    driving actions. It sums the estimated costs for each package independently,
    ignoring vehicle capacity and shared trips (relaxation).

    # Assumptions
    - The road network is static and bidirectional (if road A-B exists, B-A also exists).
    - Vehicles can traverse any road.
    - Package size and vehicle capacity constraints are ignored (relaxation).
    - A vehicle is always available to pick up/drop a package at its location.
    - All locations mentioned in the problem are part of the road network graph or are reachable.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from static `road` facts.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its final goal location:
    1. Determine the package's current effective location. This is either its
       direct location if `(at package location)` is true, or the location
       of the vehicle it is currently `(in package vehicle)`.
    2. Check if the package is already `(at package goal_location)`. If yes,
       the cost for this package is 0.
    3. If the package is not `(at package goal_location)`:
       - Find the shortest path distance `dist` between the package's current
         effective location and its goal location using the precomputed distances.
       - If the package is currently `(in package vehicle)`:
         - Add `dist` (for vehicle drive actions) + 1 (for drop action) to the total heuristic.
         - If `dist` is infinite (unreachable) or location is unknown, add a base cost of 2 (representing at least one drive and one drop).
       - If the package is currently `(at package current_location)` (not in a vehicle):
         - Add 1 (for pick-up action) + `dist` (for vehicle drive actions) + 1 (for drop action)
           to the total heuristic.
         - If `dist` is infinite (unreachable) or location is unknown, add a base cost of 3 (representing at least one pick, one drive, and one drop).
    4. Sum the costs calculated for each package. The total sum is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and precomputing shortest path distances.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2: # Ensure correct number of arguments
                    package, location = args[0], args[1]
                    self.package_goals[package] = location
                # else: Malformed goal fact, ignore or handle error

        # Build the road network graph.
        self.graph = {}
        locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3: # Ensure correct number of arguments
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                # Assuming roads are bidirectional
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)

        # Ensure all locations mentioned in goals are in the graph keys,
        # even if they have no roads connected (though unlikely in valid problems).
        # This prevents BFS errors if a goal location isn't a key.
        for loc in locations:
             self.graph.setdefault(loc, set())
        for goal_loc in self.package_goals.values():
             self.graph.setdefault(goal_loc, set())


        # Precompute all-pairs shortest path distances.
        self.distances = {}
        # Use keys from the graph built so far to get all relevant locations
        all_locations = list(self.graph.keys())
        for start_loc in all_locations:
            self.distances[start_loc] = bfs(self.graph, start_loc)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where objects are currently located or contained.
        # current_locations[obj] = location (for packages and vehicles)
        # package_in_vehicle[package] = vehicle
        current_locations = {}
        package_in_vehicle = {}

        # First pass: Find all 'at' facts for vehicles and packages
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc

        # Second pass: Find all 'in' facts for packages and update their effective location
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 package_in_vehicle[package] = vehicle
                 # The package's effective location is the vehicle's location
                 # Only update if vehicle location is known
                 if vehicle in current_locations:
                     current_locations[package] = current_locations[vehicle]
                 # else: Vehicle location unknown, package location remains unknown for now.


        total_cost = 0  # Initialize action cost counter.

        # Consider each package that has a goal location defined.
        for package, goal_l in self.package_goals.items():
            # Check if the package is already at its goal location according to the goal predicate.
            # This is the condition that must be met for this package in the goal state.
            is_at_goal_fact = f'(at {package} {goal_l})' in state

            if not is_at_goal_fact:
                # Package is not yet satisfying the goal condition (at package goal_l).
                # Estimate cost to reach this condition.

                # Find its current effective location.
                current_l = current_locations.get(package)

                # Determine if the package is currently inside a vehicle.
                is_in_vehicle = package in package_in_vehicle

                # If the package's current location is unknown (e.g., malformed state),
                # we cannot calculate distance. Add a base penalty.
                if current_l is None:
                     # This state is likely problematic or unreachable in a standard problem.
                     # Add a significant cost to discourage exploring such states.
                     # A simple base cost representing minimum actions (pick/load + drive + drop/unload)
                     # is 3 if it needs picking up, 2 if it just needs driving/dropping.
                     # Since location is unknown, assume worst case? Or just a fixed penalty.
                     # Let's add a base cost of 3 as a fallback.
                     total_cost += 3
                     continue # Move to the next package

                # Calculate shortest distance from current location to goal location.
                # Ensure both current_l and goal_l are valid keys in the distances dictionary
                # before attempting lookup. They should be if graph building is correct.
                dist = self.distances.get(current_l, {}).get(goal_l)

                if is_in_vehicle:
                    # Package is in a vehicle at current_l.
                    # Needs vehicle to drive from current_l to goal_l, then drop.
                    if dist is not None and dist != float('inf'):
                        total_cost += dist # Drive cost (minimum 1 if dist > 0)
                        total_cost += 1    # Drop cost
                    else:
                        # Unreachable or distance lookup failed. Add a base cost (drive + drop).
                        # Minimum actions if in vehicle and not at goal: 1 drive + 1 drop = 2.
                        total_cost += 2
                else: # Package is (at p current_l)
                    # Needs pick up, vehicle drive from current_l to goal_l, then drop.
                    if dist is not None and dist != float('inf'):
                        total_cost += 1 # Pick cost
                        total_cost += dist # Drive cost (minimum 1 if dist > 0)
                        total_cost += 1 # Drop cost
                    else:
                        # Unreachable or distance lookup failed. Add a base cost (pick + drive + drop).
                        # Minimum actions if not in vehicle and not at goal: 1 pick + 1 drive + 1 drop = 3.
                        total_cost += 3

        return total_cost
