from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or invalid fact format defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move each
    package to its goal location, considering pick-up, drop, and vehicle movement.
    It relaxes the problem by ignoring vehicle capacity and assuming a vehicle
    is always available where needed. The estimated cost for a package is the
    sum of:
    - 1 (if package is on the ground, for pick-up)
    - Shortest path distance (number of drive actions) from the package's
      current effective location (its location if on ground, or its vehicle's
      location if in a vehicle) to its goal location.
    - 1 (for drop-off at the goal location).
    The total heuristic is the sum of these costs for all packages that are
    not yet at their goal location.

    # Assumptions
    - Packages must be picked up by a vehicle to be moved (if on the ground).
    - Packages must be dropped from a vehicle to be placed at a location.
    - Vehicle movement cost is 1 per road segment.
    - Capacity constraints are ignored.
    - Any vehicle can transport any package.
    - The shortest path between locations is the minimum number of drive actions.
    - Objects in `(at ?obj ?loc)` goals are packages.
    - Objects starting with 'v' in `(at ?obj ?loc)` facts are vehicles.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations based on static `road` facts.
    - Precomputes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a goal location:
    1. Check if the package is already at its goal location on the ground. If yes, cost is 0 for this package and move to the next.
    2. If not at the goal, determine the package's current effective location and whether a pick-up is needed:
       - If the package is on the ground at some location L: The effective location is L, and a pick-up action is needed (cost +1).
       - If the package is inside a vehicle V, and V is at location L': The effective location is L', and no immediate pick-up is needed (cost +0).
       - If the package exists but its location is unknown (not 'at' or 'in'), or its vehicle's location is unknown, the goal is unreachable (cost infinity).
    3. If the effective location is known and finite:
       - Calculate the shortest path distance (number of drive actions) from the effective location to the package's goal location using the precomputed distances.
       - Add this distance to the cost.
       - Add 1 for the final drop-off action at the goal location.
    4. The total heuristic value is the sum of the estimated costs for all packages.
    5. If the cost calculation for any package results in infinity (e.g., unreachable location), the total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and precomputing shortest paths.
        """
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        # Collect all objects mentioned in goals to identify potential packages
        goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.goal_locations[obj] = loc
                goal_objects.add(obj)
            # Add other goal types if they imply package existence, though 'at' is most common for packages

        # 2. Build the location graph from road facts.
        self.graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Roads are typically bidirectional

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.goal_locations.values():
             if loc not in self.graph:
                 self.graph[loc] = []
                 locations.add(loc)

        # Ensure all locations mentioned in road facts are in the graph keys
        for loc in locations:
             self.graph.setdefault(loc, [])

        # 3. Precompute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_node in self.graph:
            self.distances[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find shortest distances to all other nodes.
        Returns a dictionary {location: distance}. Unreachable locations have distance infinity.
        """
        distances = {node: float('inf') for node in self.graph}

        # Handle the case where the start_node is not in the graph keys (isolated location)
        if start_node not in self.graph:
             # If the graph is completely empty (no road facts), and this is a location,
             # distance to itself is 0. Otherwise, it's isolated from the main graph.
             # We need to check if this isolated node is relevant (e.g., a goal location or initial location)
             is_relevant_isolated_node = start_node in self.goal_locations.values() or any(match(f, "at", "*", start_node) for f in self.task.initial_state)
             if not self.graph and is_relevant_isolated_node:
                  return {start_node: 0}
             # If not in graph keys and graph is not empty, it's isolated. Distance to itself is 0.
             # Distances to other nodes in the graph remain infinity from the initial dict.
             if is_relevant_isolated_node:
                 distances[start_node] = 0 # Add the isolated node to the distances map
             return distances # No neighbors to explore from here

        # Standard BFS if the start_node is in the graph keys
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Use get with default [] for nodes that might be keys but have no neighbors
            for neighbor in self.graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations of packages and vehicles.
        package_locations = {} # {package: location} if on ground
        vehicle_locations = {} # {vehicle: location}
        package_in_vehicle = {} # {package: vehicle} if in vehicle

        # Identify all packages that have a goal location
        packages_to_move = set(self.goal_locations.keys())

        # Populate current locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume objects in goal_locations are packages, others starting with 'v' are vehicles
                if obj in packages_to_move:
                     package_locations[obj] = loc
                elif obj.startswith('v'): # Heuristic assumption based on example naming
                     vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                if pkg in packages_to_move:
                    package_in_vehicle[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that needs to reach a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # Use the match helper for robustness
            if match(f"(at {package} {goal_location})", "at", package, goal_location) in state:
                 continue # Package is already at the goal location on the ground

            # Determine the package's current effective location and required pick-up cost
            current_effective_location = None
            pick_cost = 0

            if package in package_locations:
                 # Package is on the ground
                 current_effective_location = package_locations[package]
                 pick_cost = 1 # Needs a pick-up action

            elif package in package_in_vehicle:
                 # Package is in a vehicle
                 vehicle = package_in_vehicle[package]
                 if vehicle in vehicle_locations:
                      current_effective_location = vehicle_locations[vehicle]
                      # No pick-up needed *yet*, it's already in a vehicle
                      pick_cost = 0
                 else:
                      # Vehicle location unknown - treat as unreachable
                      current_effective_location = float('inf')
            else:
                 # Package exists (has a goal) but is not 'at' a location or 'in' a vehicle.
                 # This state is likely unreachable or invalid in a standard PDDL execution.
                 # Penalize heavily.
                 current_effective_location = float('inf')


            # Calculate cost from current effective location to goal location
            if current_effective_location != float('inf'):
                # Get distance from the precomputed table
                # Handle cases where start or end location might not be in the graph (isolated locations)
                start_dist_map = self.distances.get(current_effective_location, {})
                dist = start_dist_map.get(goal_location, float('inf'))

                if dist != float('inf'):
                    # Cost for this package: pick (if needed) + drive + drop
                    total_cost += pick_cost + dist + 1
                else:
                    # Goal location is unreachable from the package's current effective location
                    total_cost = float('inf') # Propagate infinity

            else:
                 # Package location was unknown
                 total_cost = float('inf') # Propagate infinity

            # If total_cost became infinity, we can stop processing packages
            if total_cost == float('inf'):
                 break

        # Return the estimated cost. If any part was unreachable, the total is infinity.
        return total_cost
