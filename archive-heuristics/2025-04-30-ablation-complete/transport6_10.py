# Imports
from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Handle unexpected format - though state facts are expected to be strings
    return []

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_road_graph(static_facts):
    """Builds an adjacency list representation of the road network from static facts."""
    graph = {}
    for fact in static_facts:
        if match(fact, "road", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            if loc1 not in graph:
                graph[loc1] = []
            if loc2 not in graph:
                graph[loc2] = []
            # Assuming bidirectional roads based on domain example
            graph[loc1].append(loc2)
            graph[loc2].append(loc1)
    return graph

def compute_all_pairs_shortest_paths(graph):
    """Computes shortest path distances between all pairs of locations using BFS."""
    locations = list(graph.keys())
    distances = {}
    for start_loc in locations:
        distances[start_loc] = {}
        queue = deque([(start_loc, 0)])
        visited = {start_loc}
        while queue:
            current_loc, dist = queue.popleft()
            distances[start_loc][current_loc] = dist
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
    return distances

# Heuristic class definition
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, considering pickup, drop,
    and the shortest path distance (number of drive actions) between locations.
    It sums these costs independently for each package that is not yet at its goal.
    Vehicle capacity and availability are ignored in this relaxed calculation.

    # Assumptions
    - Roads are bidirectional (inferred from typical domain definitions and examples).
    - Any package can be picked up by any vehicle (ignoring capacity constraints).
    - A vehicle is always available when needed at a package's location.
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. For each package 'p' that has a goal location 'goal_l' (extracted during initialization):
        a. Determine the current status of 'p' by examining the state facts:
           - Is it at a location 'current_l' (fact `(at p current_l)` is true)?
           - Is it inside a vehicle 'v' (fact `(in p v)` is true)? If so, find the location of 'v' by looking for a fact `(at v current_l)`.
        b. If the package is currently on the ground at its goal location (`(at p goal_l)` is true), the cost for this package is 0. Continue to the next package.
        c. If the package is not at its goal location on the ground:
           - Find the package's effective current location (either where it is on the ground, or where the vehicle carrying it is). If the package's status or vehicle's location cannot be determined, assign a high penalty cost for this package and move to the next.
           - Calculate the estimated transport cost as the shortest path distance (number of 'drive' actions) between the effective current location and the goal location. Use pre-computed distances. If no path exists, assign a high penalty cost.
           - Calculate the estimated action cost:
             - If the package is currently on the ground, it needs 1 'pick-up' action and 1 'drop' action. Total action cost = 2.
             - If the package is currently inside a vehicle, it needs 1 'drop' action. Total action cost = 1.
           - The total cost for this package is the transport cost + action cost.
        d. Add the cost calculated for package 'p' to the total heuristic cost.
    3. Return the total sum as the heuristic value for the state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the road network graph."""
        self.task = task # Store task for access to goals and static facts

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.task.goals:
            # Assuming goals are always of the form (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goal predicates if any exist

        # Build the road network graph
        self.road_graph = build_road_graph(self.task.static)

        # Compute all-pairs shortest paths
        self.distances = compute_all_pairs_shortest_paths(self.road_graph)

        # Define a large penalty value for unreachable locations
        # Use a value larger than any possible path + max actions per package
        max_possible_dist = len(self.road_graph) if self.road_graph else 1000
        # Max actions per package is dist + 2 (pickup+drop)
        self.unreachable_penalty = max_possible_dist + 3 # Ensure it's larger than any valid path cost

    def get_distance(self, loc1, loc2):
        """Lookup shortest path distance between two locations, returning penalty if no path."""
        # Check if both locations are in the pre-computed distances map
        if loc1 in self.distances and loc2 in self.distances[loc1]:
             return self.distances[loc1][loc2]
        # If locations are not in the graph or not connected
        return self.unreachable_penalty

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current location for objects (packages on ground, vehicles)
        object_locations = {} # Maps object -> location (if at)
        # Track package containment
        package_carrier = {} # Maps package -> vehicle (if in)

        # Parse state facts to populate the above dictionaries
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts

             predicate = parts[0]
             if predicate == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 object_locations[obj] = loc
             elif predicate == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 package_carrier[package] = vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_location = None
            is_on_ground = False

            # Determine package's current status and effective location
            if package in object_locations:
                # Package is on the ground
                current_location = object_locations[package]
                is_on_ground = True
            elif package in package_carrier:
                # Package is in a vehicle
                vehicle = package_carrier[package]
                if vehicle in object_locations:
                    # Vehicle location is known
                    current_location = object_locations[vehicle]
                else:
                    # Vehicle location unknown - package is effectively lost or state is malformed
                    # Assign a high penalty cost for this package and move to the next.
                    # This state is likely undesirable or unreachable in a valid plan.
                    total_cost += self.unreachable_penalty
                    continue # Skip this package

            # If package location/status could not be determined (e.g., not in 'at' or 'in' facts)
            if current_location is None:
                 # This should ideally not happen in valid states
                 total_cost += self.unreachable_penalty
                 continue # Skip this package

            # Check if package is already at its goal location on the ground
            # The goal is typically (at package goal_location)
            if is_on_ground and current_location == goal_location:
                 # Package is exactly where it needs to be. Cost is 0 for this package.
                 continue

            # Package is not at its goal location on the ground. It needs actions.

            # Calculate transport cost (distance)
            # Distance is from the package's effective current location to the goal location.
            transport_cost = self.get_distance(current_location, goal_location)

            # If the goal location is unreachable from the current location, penalize heavily
            if transport_cost >= self.unreachable_penalty:
                 total_cost += self.unreachable_penalty
                 continue # Skip this package

            # Calculate action costs (pickup/drop)
            action_cost = 0
            if is_on_ground:
                # Package is on the ground, needs pickup and drop
                action_cost += 1 # pick-up
                action_cost += 1 # drop
            else: # Package is in a vehicle
                # Package is in a vehicle, needs drop
                action_cost += 1 # drop
                # If the package is in a vehicle *at* the goal location,
                # transport_cost will be 0, and action_cost is 1 (drop).
                # This correctly estimates the remaining action.

            # Total cost for this package
            total_cost += transport_cost + action_cost

        return total_cost
