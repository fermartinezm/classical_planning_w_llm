# Placeholder for the base class if not provided in the execution environment
# In a real setup, this would be imported from a planning framework.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The number of parts must match the number of args for a successful match,
    # unless args contains wildcards that effectively match multiple parts (not typical PDDL matching).
    # A simple zip and match is usually sufficient for standard PDDL predicates.
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network."""
    graph = {}
    for fact in static_facts:
        if match(fact, "road", "*", "*"):
            _, l1, l2 = get_parts(fact)
            if l1 not in graph:
                graph[l1] = set()
            if l2 not in graph:
                graph[l2] = set()
            graph[l1].add(l2)
            graph[l2].add(l1) # Assuming roads are bidirectional
    return graph

def precompute_distances(graph):
    """
    Computes shortest path distances between all pairs of locations
    in the road graph using BFS.
    """
    distances = {}
    locations = list(graph.keys())
    for start_loc in locations:
        distances[start_loc] = {}
        # Perform BFS from start_loc
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        distances[start_loc][start_loc] = 0

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc in graph: # Ensure current_loc is a valid node in the graph
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and potential
    synergies/conflicts when multiple packages need the same vehicle or path.
    The cost for a package is estimated based on its current state (on the ground
    or in a vehicle) and the shortest path distance between locations.

    # Assumptions
    - Each package needs to reach a specific goal location specified in the task goals.
    - Vehicles are available to pick up packages when needed (capacity is relaxed).
    - Roads are bidirectional (handled during graph construction).
    - The cost of each relevant action (pick-up, drop, drive) is 1.
    - The shortest path distance between locations is precomputed using BFS.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from static facts defining 'road' connections.
    - Precompute the shortest path distances between all pairs of locations found in the graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status (location or vehicle) for every package and vehicle by examining 'at' and 'in' facts in the state. Store this in a dictionary `current_status`.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through each package `p` that has a goal location `goal_loc` defined in the task goals (`self.goal_locations`).
    4. For the current package `p`:
       a. Find its current status (`current_state_info`) from the `current_status` dictionary. If the package is not found in `current_status` (which implies an invalid state), return infinity as the state is likely unreachable or malformed.
       b. Check if the package is currently at its goal location `goal_loc`. This is true if the fact '(at p goal_loc)' is present in the state. If it is, the cost for this package is 0, and we move to the next package.
       c. If the package is not at its goal location, determine its current state:
          - If `(at p current_loc)` is in the state (where `current_loc` is `current_state_info`):
             - The package is on the ground at `current_loc`.
             - It needs to be picked up (1 action).
             - It needs to be transported from `current_loc` to `goal_loc`. The minimum drive actions required is the shortest path distance `dist(current_loc, goal_loc)`.
             - It needs to be dropped at `goal_loc` (1 action).
             - The estimated cost for this package is 1 (pick) + `dist(current_loc, goal_loc)` (drive) + 1 (drop).
          - If `(in p v)` is in the state (where `v` is `current_state_info`):
             - The package is inside vehicle `v`.
             - Find the current location of vehicle `v` (`vehicle_location`) from the `current_status` dictionary.
             - If `vehicle_location` is not found in state (e.g., vehicle not at any location), this state is likely invalid or unsolvable from here. Return infinity.
             - If `vehicle_location` is not `goal_loc`:
                - The vehicle needs to be transported from `vehicle_location` to `goal_loc`. The minimum drive actions required is `dist(vehicle_location, goal_location)`.
                - The package needs to be dropped at `goal_loc` (1 action).
                - The estimated cost for this package is `dist(vehicle_location, goal_location)` (drive) + 1 (drop).
             - If `vehicle_location` is `goal_loc`:
                - The package needs to be dropped at `goal_loc` (1 action).
                - The estimated cost for this package is 1 (drop).
       d. Add the calculated estimated cost for the current package to the `total_cost`.
    5. Return the final `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road graph, and precomputing distances.
        """
        # Use the base class constructor to store goals and static facts
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Goal predicates are typically 'at'
            if predicate == "at" and len(args) == 2:
                package, location = args
                self.goal_locations[package] = location
            # We ignore other potential goal predicates if any exist and are not 'at'

        # Build the road graph from static facts.
        self.road_graph = build_road_graph(self.static)

        # Precompute shortest path distances between all locations found in the graph.
        self.distances = precompute_distances(self.road_graph)

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returning infinity if no path or location not in graph."""
        # Check if both locations are in the precomputed distances map.
        # A location might not be in the map if it didn't appear in any 'road' fact,
        # but appeared in 'at' facts in the initial state or goals.
        # For a solvable problem, all relevant locations should be connected or reachable.
        # If a location is not in the graph keys, it means it wasn't part of the road network.
        # Assuming packages/vehicles only exist at locations part of the road network for transport.
        if loc1 in self.distances and loc2 in self.distances[loc1]:
            return self.distances[loc1][loc2]
        # If a location is not in the graph or no path exists, return infinity.
        # This correctly makes states leading to disconnected locations have infinite heuristic.
        return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track where packages and vehicles are currently located or contained.
        # This dictionary will map object names (packages, vehicles) to their
        # current status (location string or vehicle name string).
        current_status = {}

        # Iterate through state facts to find 'at' and 'in' predicates
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                current_status[obj] = location

            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through each package that has a goal location specified.
        for package, goal_location in self.goal_locations.items():
            # Find the current status of the package.
            current_state_info = current_status.get(package)

            # If package is not mentioned in 'at' or 'in' facts, it's an issue, but we skip.
            if current_state_info is None:
                 # print(f"Warning: Package {package} not found in state 'at' or 'in' facts.")
                 # If a package is not in the state at all, it's likely an invalid state.
                 # Returning infinity is appropriate for states that cannot lead to a goal.
                 return float('inf')

            # Check if the package is currently at its goal location.
            # This is true if the fact '(at package goal_location)' exists in the state.
            if f"(at {package} {goal_location})" in state:
                 # Package is at the goal location. Cost for this package is 0.
                 continue

            # Package is not at its goal location. Calculate its estimated cost.
            package_cost = 0

            # Determine if the package is on the ground or in a vehicle
            is_on_ground = f"(at {package} {current_state_info})" in state
            is_in_vehicle = f"(in {package} {current_state_info})" in state

            if is_on_ground:
                current_location = current_state_info # current_state_info is the location string
                # Needs pick-up (1 action)
                package_cost += 1
                # Needs drive from current_location to goal_location
                dist = self.get_distance(current_location, goal_location)
                if dist == float('inf'): return float('inf') # Unsolvable path
                package_cost += dist
                # Needs drop (1 action)
                package_cost += 1

            elif is_in_vehicle:
                vehicle_name = current_state_info # current_state_info is the vehicle name
                # Find the location of the vehicle
                vehicle_location = current_status.get(vehicle_name)

                if vehicle_location is None:
                    # Vehicle location not found in state. Invalid state or vehicle not at a location?
                    # print(f"Warning: Vehicle {vehicle_name} containing {package} has no location in state.")
                    return float('inf') # Cannot proceed if vehicle location is unknown

                # Needs drive from vehicle_location to goal_location (if not already there)
                if vehicle_location != goal_location:
                    dist = self.get_distance(vehicle_location, goal_location)
                    if dist == float('inf'): return float('inf') # Unsolvable path
                    package_cost += dist

                # Needs drop (1 action)
                package_cost += 1

            else:
                 # This case should not happen in a valid state if the package exists,
                 # as it must be either 'at' a location or 'in' a vehicle.
                 # print(f"Warning: Package {package} state info '{current_state_info}' is neither 'at' nor 'in'.")
                 return float('inf') # Invalid state representation

            # Add the package's estimated cost to the total.
            total_cost += package_cost

        return total_cost
