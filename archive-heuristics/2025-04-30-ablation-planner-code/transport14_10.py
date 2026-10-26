from fnmatch import fnmatch
from collections import defaultdict, deque
# Assuming a base class 'Heuristic' exists in 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not available
# This allows the code structure to be correct even without the external dependency.
# In a real scenario, the actual base class would be imported.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        raise NotImplementedError("Subclass must implement abstract method")

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def build_graph(road_facts):
    """Build an adjacency list graph from road facts."""
    graph = defaultdict(set)
    locations = set()
    for fact in road_facts:
        parts = get_parts(fact)
        if len(parts) == 3: # Ensure fact has correct structure (road l1 l2)
            _, l1, l2 = parts
            graph[l1].add(l2)
            locations.add(l1)
            locations.add(l2)
    return graph, list(locations)

def compute_distances(graph, locations):
    """Compute shortest path distances between all pairs of locations using BFS."""
    distances = {}
    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0
        while q:
            current_loc, dist = q.popleft()
            # Store distance
            distances[(start_node, current_loc)] = dist

            # Explore neighbors
            for neighbor in graph.get(current_loc, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
    return distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the required number of actions to move all
    misplaced packages to their goal locations. It calculates the cost for
    each misplaced package independently, summing up the estimated actions
    (pick-up, drive, drop) needed for that specific package based on shortest
    path distances in the road network.

    # Assumptions
    - The goal is to move specific packages to specific locations (`(at package location)`).
    - The road network defined by `(road l1 l2)` facts is static.
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity and availability are ignored. Each package's transport
      is considered in isolation, assuming a vehicle is available when needed.
    - Shortest path in the road network represents the minimum number of drive actions.

    # Heuristic Initialization
    1. Parses static facts to build the road network graph.
    2. Computes all-pairs shortest path distances between locations using BFS.
    3. Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containment status for every package
       that has a goal location. A package can be `(at package location)`
       or `(in package vehicle)`. If it's in a vehicle, find the vehicle's
       location `(at vehicle location)`.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` with a goal location `l_goal`:
       a. Determine the package's current physical location `l_current`.
          - If `(at p l_current)` is in the state, `l_current` is its location.
          - If `(in p v)` is in the state, find `(at v l_v)` in the state; `l_current` is `l_v`.
       b. If the package is already at its goal location (`l_current == l_goal` and it's on the ground, i.e., `(at p l_goal)` is in state), add 0 to the total cost for this package.
       c. If the package is misplaced:
          - If the package is on the ground at `l_current` (`(at p l_current)` in state):
            The estimated cost for this package is 1 (pick-up) + shortest_distance(`l_current`, `l_goal`) + 1 (drop). Add this to the total cost.
          - If the package is inside a vehicle `v` which is at `l_v` (`(in p v)` and `(at v l_v)` in state):
            The estimated cost for this package is shortest_distance(`l_v`, `l_goal`) + 1 (drop). Add this to the total cost.
          - If the goal location is unreachable from the package's current location (or vehicle's location), the heuristic returns infinity.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and storing goal locations.

        Args:
            task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Precompute road network distances
        road_facts = [fact for fact in static_facts if match(fact, "road", "*", "*")]
        self.location_graph, self.locations = build_graph(road_facts)
        self.distances = compute_distances(self.location_graph, self.locations)

        # 2. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is typically (at package location)
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "at":
                 _, package, location = parts
                 self.goal_locations[package] = location
            # Note: This heuristic assumes goals are only (at package location).
            # If other goal types existed (e.g., (in package vehicle)), this would need extension.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        Args:
            node: The current search node containing the state.

        Returns:
            An estimated cost (integer) or float('inf') if the goal is unreachable.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Track current location/status of packages and vehicles
        package_current_info = {} # package -> {'type': 'at'/'in', 'loc': location/vehicle}
        vehicle_locations = {} # vehicle -> location

        # First pass: Identify package status (at or in) and vehicle locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # If this object is a package we care about (has a goal)
                if obj in self.goal_locations:
                     package_current_info[obj] = {'type': 'at', 'loc': loc}
                # Assume anything else 'at' a location is a vehicle
                # This is a simplification, but works for typical transport problems
                # where only vehicles and goal packages are locatable.
                elif obj in self.locations or loc in self.locations: # Basic check if obj/loc are locations/locatables
                     vehicle_locations[obj] = loc # Assume it's a vehicle

            elif parts[0] == 'in' and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 if package in self.goal_locations:
                     package_current_info[package] = {'type': 'in', 'loc': vehicle}

        total_cost = 0

        # Calculate cost for each misplaced package
        for package, goal_location in self.goal_locations.items():
            current_info = package_current_info.get(package)

            # If package info is missing, it might mean it's not in the state facts.
            # This could happen if it's already at the goal and the fact was removed
            # by a previous action (though actions typically add the new 'at' fact).
            # A robust check is to see if the goal fact is explicitly in the state.
            if f'(at {package} {goal_location})' in state:
                continue # Package is already at its goal location on the ground

            # Package is not at its goal location on the ground. It's either misplaced
            # on the ground or inside a vehicle.
            if current_info is None:
                 # Package is not found in 'at' or 'in' facts. This indicates an
                 # invalid state representation or a package that doesn't exist
                 # in the current state facts but is in the goal.
                 # Assuming valid states, this case implies the package is missing.
                 # Treat as unreachable for safety in greedy search.
                 return float('inf')

            current_type = current_info['type']
            current_loc_or_vehicle = current_info['loc']

            if current_type == 'at':
                # Package is on the ground at current_loc_or_vehicle
                l_current = current_loc_or_vehicle
                # We already checked if it's at the goal location on the ground above.
                # So if we are here, it's misplaced on the ground.
                # Cost: pick-up (1) + drive (distance) + drop (1)
                dist = self.distances.get((l_current, goal_location))
                if dist is None: # No path found
                    return float('inf')
                total_cost += 1 + dist + 1 # pick + drive + drop

            elif current_type == 'in':
                # Package is inside a vehicle (current_loc_or_vehicle is the vehicle name)
                vehicle = current_loc_or_vehicle
                # Find vehicle's location
                l_vehicle = vehicle_locations.get(vehicle)
                if l_vehicle is None:
                    # Vehicle location not found? Invalid state or parsing error.
                    # A vehicle carrying a package must be at a location.
                    return float('inf') # Vehicle carrying package has no location?

                # Cost: drive (distance) + drop (1)
                dist = self.distances.get((l_vehicle, goal_location))
                if dist is None: # No path found
                    return float('inf')
                total_cost += dist + 1 # drive + drop

        # The heuristic is 0 only if the loop finishes without adding cost,
        # which happens only if all goal packages were found to be at their
        # goal locations on the ground. This satisfies the requirement that
        # h=0 only for goal states.

        return total_cost

# Example usage (assuming 'task' and 'node' objects are available):
# task = ... # Load task from PDDL files
# heuristic = transportHeuristic(task)
# state = ... # Get current state from search node
# node = type('Node', (object,), {'state': state})() # Mock node object
# h_value = heuristic(node)
# print(f"Heuristic value: {h_value}")
