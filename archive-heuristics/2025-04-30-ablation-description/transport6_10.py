from fnmatch import fnmatch
from collections import deque
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Utility functions
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing, for each package not at its goal location,
    the minimum number of actions required to get it there, ignoring vehicle capacity constraints and
    the need for vehicles to become available. The cost includes pick-up (if on ground), driving (shortest path),
    and drop-off actions.

    # Assumptions
    - Roads are bidirectional (as observed in example instances).
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored.
    - Vehicle availability is ignored (assumes a vehicle is always available when needed).
    - The heuristic calculates the sum of costs for each package independently.

    # Heuristic Initialization
    - Precomputes the shortest path distances between all pairs of locations based on the 'road' facts.
    - Stores the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each package goal (e.g., `(at packageX locationY)`) specified in the task:
       a. Check if this goal fact is already true in the current state. If yes, this package is at its goal location on the ground, and no further cost is added for this goal. Continue to the next goal.
       b. If the goal fact is not true, the package is not yet at its goal location.
       c. Determine the package's current status:
          - Find if the package is `(at packageX current_location)` or `(in packageX vehicleZ)`.
       d. If the package is `(at packageX current_location)` (on the ground) and `current_location` is not the goal location:
          - The package needs to be picked up (1 action).
          - It needs to be transported from `current_location` to `goal_location`. The minimum number of drive actions is the shortest path distance between these locations.
          - It needs to be dropped at `goal_location` (1 action).
          - Add 1 (pick) + shortest_path_distance(current_location, goal_location) + 1 (drop) to the total cost.
          - If the goal location is unreachable from the current location, the heuristic is infinite.
       e. If the package is `(in packageX vehicleZ)`:
          - Find the current location of `vehicleZ` (`(at vehicleZ vehicle_location)`).
          - The package needs to be transported from `vehicle_location` to `goal_location`. The minimum number of drive actions is the shortest path distance between these locations.
          - It needs to be dropped at `goal_location` (1 action).
          - Add shortest_path_distance(vehicle_location, goal_location) + 1 (drop) to the total cost.
          - If the goal location is unreachable from the vehicle's location, the heuristic is infinite.
    3. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and storing goals.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of facts)
        static_facts = task.static  # Static facts (frozenset of facts)

        # 1. Precompute shortest path distances between locations
        road_graph = {}
        locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                road_graph.setdefault(l1, set()).add(l2)
                road_graph.setdefault(l2, set()).add(l1) # Assuming roads are bidirectional based on examples

        self.distance = {}
        for start_loc in locations:
            self.distance[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distance[start_loc][start_loc] = 0

            while q:
                current_loc, dist = q.popleft()

                # Get neighbors, handle locations with no roads
                neighbors = road_graph.get(current_loc, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distance[start_loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # 2. Store goal locations for each package
        self.goal_locations = {}
        for goal_fact in self.goals:
            goal_parts = get_parts(goal_fact)
            if goal_parts[0] == "at":
                package, location = goal_parts[1], goal_parts[2]
                self.goal_locations[package] = location
            # Ignore other potential goal types if any

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.

        @param node: The current search node, containing the state.
        @return: The estimated heuristic cost (integer or float('inf')).
        """
        state = node.state

        # Map objects (packages, vehicles) to their current location (or vehicle if inside)
        current_locations = {}
        # Keep track of which packages are inside vehicles
        packages_in_vehicles = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_locations[package] = vehicle # Store the vehicle name
                packages_in_vehicles.add(package)

        total_cost = 0

        # Iterate through the goal facts to find unmet goals
        for goal_fact in self.goals:
            # We only care about (at package location) goals for this heuristic
            goal_parts = get_parts(goal_fact)
            if goal_parts[0] != "at":
                continue # Ignore other goal types

            package, goal_loc = goal_parts[1], goal_parts[2]

            # Check if the goal (at package goal_loc) is already met
            if goal_fact in state:
                continue # Goal met for this package, cost is 0 for this part

            # Goal is not met. Calculate cost for this package.
            current_state_info = current_locations.get(package)

            if current_state_info is None:
                # This package is not mentioned in 'at' or 'in' facts. Should not happen in valid PDDL states.
                # Treat as unreachable.
                return float('inf')

            current_loc = None # This will be the effective location for distance calculation

            if package in packages_in_vehicles:
                # Package is inside a vehicle
                vehicle = current_state_info
                vehicle_loc = current_locations.get(vehicle)
                if vehicle_loc is None:
                    # Vehicle carrying package is not at any location. Should not happen.
                    return float('inf')
                current_loc = vehicle_loc

                # Cost to move vehicle from current_loc to goal_loc + cost to drop
                # Distance lookup: self.distance[from_loc][to_loc]
                # Use .get() with default float('inf') to handle cases where a location might not be in the precomputed map
                # (e.g., if it was in the initial state but not connected by roads to anything else mentioned).
                drive_cost = self.distance.get(current_loc, {}).get(goal_loc, float('inf'))
                drop_cost = 1

                # If goal_loc is unreachable from current_loc, heuristic is infinity
                if drive_cost == float('inf'):
                     return float('inf')

                total_cost += drive_cost + drop_cost

            else:
                # Package is on the ground
                current_loc = current_state_info # This is the location on the ground

                # Cost to pick up + cost to move from current_loc to goal_loc + cost to drop
                # Distance lookup: self.distance[from_loc][to_loc]
                drive_cost = self.distance.get(current_loc, {}).get(goal_loc, float('inf'))
                pick_cost = 1
                drop_cost = 1

                # If goal_loc is unreachable from current_loc, heuristic is infinity
                if drive_cost == float('inf'):
                     return float('inf')

                total_cost += pick_cost + drive_cost + drop_cost

        # Return the total estimated cost
        return total_cost
