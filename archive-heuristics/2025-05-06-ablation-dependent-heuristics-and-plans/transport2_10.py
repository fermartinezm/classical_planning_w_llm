# Assuming Heuristic base class is available from a separate file/module
# For demonstration purposes, a dummy class is used if the base class file is not provided.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy Heuristic class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError("Heuristic not implemented")

from fnmatch import fnmatch
from collections import deque
# import sys # Not strictly needed as float('inf') is built-in

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts is sufficient to match the pattern
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the costs for all
    packages. It considers the cost of picking up, dropping off, and driving,
    based on shortest path distances in the road network.

    # Assumptions
    - Each package needs to reach a specific goal location defined by an `(at ?p ?l)` goal fact.
    - Vehicles are assumed to be available to pick up packages on the ground when needed, without considering their current location or the cost to reach the package.
    - Vehicle capacity constraints are ignored.
    - The cost of driving between locations is the shortest path distance in the road network (each road segment costs 1 drive action).
    - Each pick-up and drop action costs 1.
    - All locations mentioned in the problem (initial state, goals, roads) are part of a single connected road network, or unreachable goals are handled by returning infinity.
    - Vehicle object names start with 'v'.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations connected by bidirectional roads from static facts.
    - Computes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package and the location of every vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a defined goal location:
       a. Determine the package's current status:
          - On the ground at location `current_l` (found via `(at ?p ?l)` fact).
          - Inside a vehicle `v` (found via `(in ?p ?v)` fact).
       b. If the package is inside a vehicle `v`, find the vehicle's current location `vehicle_l` (found via `(at ?v ?l)` fact).
       c. Determine the package's goal location `goal_l` (from initialization).
       d. If the package is on the ground at `current_l` and `current_l` is the same as `goal_l`, this package is done; continue to the next package.
       e. Calculate the estimated cost for this package:
          - If the package is on the ground at `current_l` (and `current_l != goal_l`):
            - Cost = 1 (pick-up) + shortest_distance(`current_l`, `goal_l`) (drive) + 1 (drop).
          - If the package is inside vehicle `v` at `vehicle_l`:
            - Cost = shortest_distance(`vehicle_l`, `goal_l`) (drive) + 1 (drop).
       f. If the calculated cost for this package is infinite (due to unreachable goal location), the total heuristic is infinite.
       g. Otherwise, add this estimated cost to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest path distances between locations.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at ?pkg ?loc) goal
                package, location = args
                self.goal_locations[package] = location
            # Ignore other potential goal types if any, assuming package locations are the main goal.

        # Build the road graph and compute all-pairs shortest paths.
        self.road_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are typically bidirectional

        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_loc, all_locations):
        """
        Performs BFS from a start location to find distances to all other locations.
        Returns a dictionary mapping location to distance.
        """
        distances_from_start = {loc: float('inf') for loc in all_locations}
        distances_from_start[start_loc] = 0
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()
            current_dist = distances_from_start[current_loc]

            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances_from_start[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances_from_start


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where locatables (packages, vehicles) are currently located.
        # A package can be at a location or in a vehicle.
        # A vehicle is always at a location.
        current_locations = {} # {locatable_name: location_name_or_vehicle_name}
        vehicle_locations = {} # {vehicle_name: location_name}

        # Populate current_locations and vehicle_locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Assuming vehicles start with 'v' based on domain examples
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc

            elif predicate == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 current_locations[pkg] = veh # Package is inside a vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # If package is not mentioned in the state (shouldn't happen in valid states), skip.
            if package not in current_locations:
                 # This package's state is unknown, cannot estimate cost.
                 # For a greedy search, this might indicate an issue or unreachable goal.
                 # Returning infinity is a safe approach.
                 return float('inf')

            current_status = current_locations[package]

            # Check if the package is already at its goal location on the ground.
            # If it's in a vehicle at the goal location, it still needs to be dropped.
            is_on_ground = current_status in self.distances # Check if status is a known location key

            if is_on_ground and current_status == goal_location:
                # Package is on the ground at the goal location. No cost for this package.
                continue

            # Package is not at the goal location on the ground. Calculate cost.
            package_cost = 0
            if is_on_ground:
                # Package is on the ground at current_status (which is a location).
                current_l = current_status
                # Cost: pick-up + drive + drop
                # Assumes a vehicle is available at current_l.
                # Get distance from current location to goal location.
                # Use .get with infinity as default for safety, though graph should cover all relevant locs.
                drive_cost = self.distances.get(current_l, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location.
                    # This part of the problem might be unsolvable from this state.
                    # Return infinity.
                    return float('inf')

                package_cost = 1 + drive_cost + 1 # pick-up + drive + drop

            else:
                # Package is inside a vehicle (current_status is a vehicle name).
                vehicle_name = current_status
                # Find the vehicle's location.
                if vehicle_name not in vehicle_locations:
                     # Vehicle location not found in state? Should not happen in valid state.
                     # Treat this package's goal as unreachable from here.
                     return float('inf')

                vehicle_l = vehicle_locations[vehicle_name]

                # Cost: drive + drop
                # Vehicle drives from vehicle_l to goal_location.
                drive_cost = self.distances.get(vehicle_l, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     # Goal is unreachable from vehicle's current location.
                     return float('inf')

                package_cost = drive_cost + 1 # drive + drop

            total_cost += package_cost

        return total_cost
