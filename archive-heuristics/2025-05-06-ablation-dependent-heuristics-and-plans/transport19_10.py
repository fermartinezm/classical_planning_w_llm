from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

# BFS implementation
def bfs_shortest_paths(graph, start_node):
    """
    Computes shortest path distances from start_node to all other nodes
    in an unweighted graph using BFS.

    Args:
        graph: Adjacency list representation {node: [neighbor1, neighbor2, ...]}
        start_node: The starting node for BFS.

    Returns:
        A dictionary {node: distance} containing shortest distances.
    """
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
        # Start node is not in the graph (e.g., isolated location not in road facts)
        # It can only reach itself with distance 0.
        if start_node in distances:
             distances[start_node] = 0
        return distances # Cannot traverse from here

    distances[start_node] = 0
    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)

        # Ensure current_node is still valid in graph keys during traversal
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, independently. It sums the
    estimated costs for all packages that are not yet at their goal. The cost
    for a single package is estimated as the sum of:
    - 1 pick-up action (if the package is on the ground).
    - The shortest path distance (number of drive actions) for a vehicle to move
      from the package's current location (or the vehicle's location if the
      package is inside a vehicle) to the package's goal location.
    - 1 drop action (if the package needs to be moved).

    # Assumptions
    - Goals are always of the form (at package location).
    - Every location relevant to packages and their goals is reachable from other relevant locations in solvable problems.
    - Vehicle capacity constraints are ignored when estimating the cost for
      individual packages.
    - Any package not at its goal location needs to be moved.
    - Vehicles are always located at some location if they exist in the state.
    - Objects starting with 'p' are packages, 'v' are vehicles, 'l' are locations.

    # Heuristic Initialization
    - Extracts all road connections to build a graph of locations.
    - Collects all locations mentioned in static facts (roads) and the initial state.
    - Computes the shortest path distance between all pairs of collected locations using BFS.
    - Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic value is 0.
    2. Identify the goal location for each package that needs to be moved (i.e.,
       packages mentioned in the goal state).
    3. For each such package:
       a. Check if the package is already at its goal location in the current state
          using the exact goal fact `(at package goal_location)`. If yes, the cost
          for this package is 0.
       b. If not at the goal, determine the package's current status by examining
          the state facts:
          - Is it on the ground at some location `L_p`? (Look for `(at package L_p)`)
          - Is it inside a vehicle `V`? (Look for `(in package V)`)
       c. If the package is on the ground at `L_p`:
          - The estimated cost for this package is 1 (pick-up) + shortest_path(`L_p`, `goal_location`) + 1 (drop).
       d. If the package is inside a vehicle `V`:
          - Find the current location of vehicle `V`, say `L_v`, by looking for `(at V L_v)` in the state.
          - The estimated cost for this package is shortest_path(`L_v`, `goal_location`) + 1 (drop).
          - Note: If the package is in a vehicle *at* its goal location, the shortest path is 0, and the cost is 1 (drop).
       e. If the shortest path distance required is infinite (meaning the goal location is unreachable from the current location/vehicle location), the state is likely unsolvable or a dead end, and the heuristic returns infinity.
       f. If a package mentioned in the goal is not found in the state (neither 'at' nor 'in'), the state is likely unsolvable, and the heuristic returns infinity.
    4. Sum the estimated costs for all packages that are not at their goal location.
    5. The total sum is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (`road` relationships).
        - Precompute shortest paths between locations.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Assume objects starting with 'p' are packages and 'l' are locations
                if package.startswith('p') and location.startswith('l'):
                     self.goal_locations[package] = location

        # Build the location graph from static road facts and collect all relevant locations.
        self.location_graph = {}
        all_locations_set = set()

        # Collect locations from road facts and build graph
        for fact in self.static:
             predicate, *args = get_parts(fact)
             if predicate == "road" and len(args) == 2:
                 l1, l2 = args
                 # Assume objects starting with 'l' are locations
                 if l1.startswith('l') and l2.startswith('l'):
                     all_locations_set.add(l1)
                     all_locations_set.add(l2)
                     self.location_graph.setdefault(l1, []).append(l2)
                     self.location_graph.setdefault(l2, []).append(l1) # Assuming bidirectional

        # Collect locations from initial state 'at' facts
        for fact in self.task.initial_state: # Access initial_state via self.task
             predicate, *args = get_parts(fact)
             if predicate == "at" and len(args) == 2:
                 obj, loc = args
                 # Assume objects starting with 'l' are locations
                 if loc.startswith('l'):
                     all_locations_set.add(loc)
                     self.location_graph.setdefault(loc, []) # Ensure location exists in graph even if isolated

        # Collect locations from goal state 'at' facts
        for loc in self.goal_locations.values():
             if loc.startswith('l'): # Defensive check
                 all_locations_set.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure location exists in graph even if isolated

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in all_locations_set:
             self.shortest_paths[start_loc] = bfs_shortest_paths(self.location_graph, start_loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        # The goal is a set of facts. Check if the state contains all goal facts.
        if self.task.goals.issubset(state):
             return 0

        # Track where packages and vehicles are currently located or contained.
        current_package_status = {} # package -> (location_or_vehicle, is_in_vehicle)
        current_vehicle_locations = {} # vehicle -> location

        # Populate status dictionaries from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume objects starting with 'p' are packages, 'v' are vehicles, 'l' are locations
                if obj.startswith('p') and loc.startswith('l'):
                    current_package_status[obj] = (loc, False) # (location, on_ground)
                elif obj.startswith('v') and loc.startswith('l'):
                    current_vehicle_locations[obj] = loc # vehicle -> location
            elif predicate == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 if package.startswith('p') and vehicle.startswith('v'):
                      current_package_status[package] = (vehicle, True) # (vehicle, in_vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location
            is_at_goal_fact = f"(at {package} {goal_location})"
            if is_at_goal_fact in state:
                 continue # Package is already at its goal location

            # Package is not at its goal location, calculate cost to move it

            if package not in current_package_status:
                 # Package is not located anywhere or in any vehicle. This state is likely unsolvable.
                 # Return infinity.
                 return float('inf')

            current_status, is_in_vehicle = current_package_status[package]

            if not is_in_vehicle: # Package is on the ground at current_status (a location)
                current_loc = current_status
                # Cost: pick-up + drive + drop
                # Need shortest path from current_loc to goal_location
                # Use .get() with default float('inf') for robustness if a location isn't in shortest_paths (shouldn't happen with current init)
                drive_cost = self.shortest_paths.get(current_loc, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location.
                    return float('inf')

                cost_for_package = 1 + drive_cost + 1 # pick + drive + drop
                total_cost += cost_for_package

            else: # Package is inside a vehicle (current_status is the vehicle name)
                vehicle = current_status
                # Need the location of the vehicle
                if vehicle not in current_vehicle_locations:
                     # Vehicle containing package is not located anywhere? Invalid state.
                     return float('inf')

                current_loc = current_vehicle_locations[vehicle]
                # Cost: drive + drop
                # Need shortest path from vehicle's current_loc to goal_location
                drive_cost = self.shortest_paths.get(current_loc, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable from vehicle's current location.
                     return float('inf')

                cost_for_package = drive_cost + 1 # drive + drop
                total_cost += cost_for_package

        return total_cost
