# Assuming heuristics.heuristic_base.Heuristic is available
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise error
        # For robustness, return empty list or handle appropriately
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
    # Pattern must not be longer than the fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently. The cost for a package includes picking it up (if on the ground),
    driving it to the goal location, and dropping it. Driving cost is estimated
    using shortest path distances on the road network.

    # Assumptions
    - The road network is static and given by 'road' facts.
    - Shortest path distances on the road network represent the minimum number of 'drive' actions.
    - Vehicle availability and capacity constraints are relaxed (ignored). It is assumed that a vehicle is available with sufficient capacity whenever a package needs to be picked up or dropped.
    - Each package transport is considered independently.
    - Object names starting with 'p' are packages, and those starting with 'v' are vehicles.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph representation of the road network from 'road' facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic value is 0.
    2. Identify the current location/status of every package (on the ground at a location, or inside a vehicle). Also, identify the current location of every vehicle.
    3. Initialize the total heuristic cost to 0.
    4. For each package that has a specified goal location:
       a. Check if the package is already at its goal location on the ground. If yes, this package contributes 0 to the heuristic.
       b. If the package is not at its goal location on the ground, determine its effective current location:
          - If the package is on the ground at `current_l`, the effective location is `current_l`.
          - If the package is inside a vehicle `v`, find the current location of vehicle `v` (`vehicle_l`). The effective location is `vehicle_l`.
       c. Determine the package's goal location (`goal_l`).
       d. Calculate the estimated driving cost as the shortest path distance between the effective current location and the goal location using the precomputed distances. If the goal is unreachable, the total heuristic should be infinity.
       e. Calculate the estimated action cost (pick-up/drop):
          - If the package is currently on the ground: It needs a pick-up action and a drop action (2 actions).
          - If the package is currently inside a vehicle: It needs a drop action (1 action).
          - Note: If the package is in a vehicle *at* the goal location, the drive cost is 0, and the action cost is 1 (drop), resulting in a total of 1 for this package, which is correct.
       f. Add the driving cost and the action cost for this package to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and precomputing shortest path distances.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming the first argument of 'at' in a goal is a package
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location

        # Build the road network graph.
        self.adj = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                if len(parts) == 3:
                    _, l1, l2 = parts # Correct unpacking: parts[0] is 'road', parts[1] is l1, parts[2] is l2
                    locations.add(l1)
                    locations.add(l2)
                    if l1 not in self.adj:
                        self.adj[l1] = []
                    if l2 not in self.adj:
                        self.adj[l2] = []
                    # Add edges for both directions as roads are typically bidirectional
                    self.adj[l1].append(l2)
                    self.adj[l2].append(l1)

        self.locations = list(locations) # List of all unique locations

        # Precompute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_node in self.locations:
            self.shortest_paths[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary mapping location to distance.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in distances:
             # Start node might not be in the locations set if it was only mentioned in init/goal, not roads
             # This case indicates an issue with the problem definition or parsing, but handle defensively
             return {} # Cannot compute paths from an unknown location

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Check if u exists in adj, handles locations with no roads defined
            if u in self.adj:
                for v in self.adj[u]:
                    if v in distances and distances[v] == float('inf'): # Ensure v is a known location
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # If the current state is the goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Map locatables (packages, vehicles) to their current status (location or container).
        current_locatables_status = {} # {obj: location or vehicle}
        vehicle_locations = {} # {vehicle: location}

        # Populate current_locatables_status and vehicle_locations from the state facts.
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - locatable ?v - location)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_locatables_status[obj] = loc
                    # Assuming objects starting with 'v' are vehicles based on domain examples
                    if obj.startswith('v'):
                         vehicle_locations[obj] = loc
            elif predicate == "in":
                # (in ?x - package ?v - vehicle)
                if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_locatables_status[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that have a specified goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            # This covers cases where the package is already delivered.
            if (f"(at {package} {goal_location})") in state:
                continue # This package is already at its goal, contributes 0 cost.

            # Package is not yet at its goal location on the ground.
            # Determine the package's effective current location (where it is, or where its vehicle is).
            pkg_current_status = current_locatables_status.get(package)

            # If package status is unknown, it might not exist or be in an unexpected state.
            # For a heuristic, we can treat this as an impossible goal or skip.
            # Assuming valid problem states where packages always have an 'at' or 'in' status.
            if pkg_current_status is None:
                 # This package's goal is likely unreachable or the state is malformed.
                 # Returning infinity is appropriate for an unsolvable state.
                 return float('inf') # Cannot find package status

            # Check if the package status indicates it's inside a vehicle.
            # Assuming vehicle names start with 'v'.
            is_in_vehicle = pkg_current_status.startswith('v')

            effective_current_location = None
            action_cost = 0 # Cost for pick-up/drop actions for this package

            if is_in_vehicle:
                vehicle = pkg_current_status
                # Get the vehicle's physical location.
                vehicle_l = vehicle_locations.get(vehicle)
                if vehicle_l is None:
                    # Vehicle location is unknown - indicates a problem state issue.
                    # Treat this package's goal as unreachable.
                    return float('inf') # Cannot transport package if vehicle location is unknown.

                effective_current_location = vehicle_l
                action_cost = 1 # Package is in vehicle, only needs 1 drop action at the goal.
            else: # Package is on the ground at pkg_current_status
                effective_current_location = pkg_current_status # Use the status directly as it's the location
                action_cost = 2 # Package is on the ground, needs 1 pick-up and 1 drop action.

            # Calculate driving cost from the effective current location to the goal location.
            # Check if both locations are in our precomputed shortest paths.
            if effective_current_location not in self.shortest_paths or goal_location not in self.shortest_paths.get(effective_current_location, {}):
                 # This means the goal location is unreachable from the package's current location.
                 # The problem is likely unsolvable from this state.
                 return float('inf')

            drive_cost = self.shortest_paths[effective_current_location][goal_location]

            # If drive_cost is infinity, the goal is unreachable from here.
            if drive_cost == float('inf'):
                 return float('inf')

            # Add the cost for this package: driving cost + pick-up/drop actions.
            total_cost += drive_cost + action_cost

        return total_cost
