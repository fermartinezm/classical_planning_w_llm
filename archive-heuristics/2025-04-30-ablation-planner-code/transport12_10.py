from fnmatch import fnmatch
from collections import deque
import sys

# Assume Heuristic base class exists as in examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if the actual one is not provided
# This allows the code to be runnable standalone for syntax check,
# but the final submission should assume the planner provides it.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Add other potential task attributes needed by heuristics
            # self.objects = task.objects
            # self.initial_state = task.initial_state
            pass

        def __call__(self, node):
            raise NotImplementedError("Heuristic must implement __call__")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
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
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the necessary pick-up and drop actions for
    each package not yet at its goal and adds the maximum shortest path distance
    required for any package's transport. It ignores vehicle capacity constraints
    and vehicle availability beyond their initial locations.

    # Assumptions
    - Packages need to be picked up if on the ground and not at the goal.
    - Packages need to be dropped if in a vehicle and the vehicle is at the goal.
    - Transport involves driving a vehicle along roads.
    - Vehicle capacity is sufficient for any required pick-up (relaxation).
    - A suitable vehicle is available for pick-up/transport (relaxation).
    - The cost of driving is the shortest path distance in the road network.
    - Roads are bidirectional.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations connected by roads from static facts.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize `total_pickup_cost = 0`, `total_drop_cost = 0`, `max_drive_cost = 0`.
    2. Check if the current state is the goal state. If yes, return 0.
    3. For each package `p` that has a goal location `l_goal` defined in the task goals:
       a. Find the current state of package `p` by searching the state facts:
          - Is it on the ground at `l_current` (`(at p l_current)`)?
          - Is it inside a vehicle `v` (`(in p v)`)? If so, find the vehicle's location `l_current_v` (`(at v l_current_v)`).
       b. If package `p` is already at `l_goal` on the ground (`(at p l_goal)` is in state), this package requires 0 further actions towards its goal. Continue to the next package.
       c. If package `p` is on the ground at `l_current` where `l_current != l_goal`:
          - It needs to be picked up: Add 1 to `total_pickup_cost`.
          - It needs to be dropped at the goal: Add 1 to `total_drop_cost`.
          - It needs transport from `l_current` to `l_goal`: Find the shortest path distance `d` between `l_current` and `l_goal` using the precomputed distances. Update `max_drive_cost = max(max_drive_cost, d)`. If the goal is unreachable, the heuristic is infinity.
       d. If package `p` is inside a vehicle `v` which is at `l_current_v`:
          - It needs to be dropped at the goal: Add 1 to `total_drop_cost`.
          - If `l_current_v != l_goal`:
             - It needs transport from `l_current_v` to `l_goal`: Find the shortest path distance `d` between `l_current_v` and `l_goal` using the precomputed distances. Update `max_drive_cost = max(max_drive_cost, d)`. If the goal is unreachable, the heuristic is infinity.
    4. The total heuristic value is `total_pickup_cost + total_drop_cost + max_drive_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the road
        graph, and precomputing shortest paths.
        """
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location
                else:
                    # Handle unexpected goal format
                    print(f"Warning: Unexpected goal format: {goal}")


        # Build the road graph and collect all unique locations.
        self.road_graph = {}
        all_locations = set()

        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Roads are bidirectional
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = []
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = []
                self.road_graph[loc1].append(loc2)
                self.road_graph[loc2].append(loc1)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add any locations from goals that might not be in road facts (isolated)
        for loc in self.goal_locations.values():
             if loc not in self.road_graph:
                 self.road_graph[loc] = [] # Add as a node, even if no roads
             all_locations.add(loc)

        # Precompute shortest path distances between all pairs of locations.
        self.shortest_paths = {}
        # Ensure BFS is run for every location that could potentially be a start or end point
        # (i.e., all locations identified from roads and goals).
        for start_loc in all_locations:
             # Ensure start_loc is a key in the graph, even if it has no roads
             if start_loc not in self.road_graph:
                 self.road_graph[start_loc] = []
             self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find shortest paths to all reachable nodes."""
        # Initialize distances for all nodes known in the graph
        distances = {node: float('inf') for node in self.road_graph}
        if start_node in distances: # Check if start_node is in our graph nodes
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # Ensure current_node is a key in the graph before iterating neighbors
                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_node] + 1
                            queue.append(neighbor)

            # Store computed distances from start_node
            for end_node, dist in distances.items():
                 self.shortest_paths[(start_node, end_node)] = dist
        else:
             # If start_node wasn't in the graph nodes (e.g., an object is at an unknown location)
             # This case indicates an issue with state representation or problem definition,
             # but for robustness, we can populate paths assuming infinity to all known nodes.
             for end_node in self.road_graph:
                  self.shortest_paths[(start_node, end_node)] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of fact strings.

        # Check if the state is a goal state first for efficiency and correctness (h=0 at goal)
        is_goal_state = True
        for goal in self.goals:
             if goal not in state:
                 is_goal_state = False
                 break

        if is_goal_state:
            return 0 # Heuristic is 0 for goal states

        # Map current locations of locatables (packages and vehicles)
        # and track which packages are inside which vehicles.
        current_locations = {} # obj -> location or vehicle
        packages_in_vehicles = {} # package -> vehicle
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
                # Infer vehicle vs package based on common usage or naming convention
                # A more robust way would use task.objects and types, but let's use simple check
                # Vehicles are typically the objects that have 'at' facts and carry things.
                # Packages are objects that have 'at' or 'in' facts.
                # Assuming objects starting with 'v' are vehicles and 'p' are packages based on examples.
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
                # Packages starting with 'p' are handled by current_locations if on the ground.
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                packages_in_vehicles[package] = vehicle
                current_locations[package] = vehicle # Store vehicle name as location for 'in' facts

        total_pickup_cost = 0
        total_drop_cost = 0
        max_drive_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If a package is not mentioned in the current state facts ('at' or 'in'),
            # it's an unexpected situation in a valid PDDL state. We'll skip it.
            if package not in current_locations:
                 # print(f"Warning: Package {package} with goal {goal_location} not found in state.")
                 continue # Cannot compute cost for a package not in the state

            current_pos = current_locations[package] # This is either a location string or a vehicle string

            # Check if the package is already at its goal location on the ground.
            # This check is implicitly covered by the logic below, but explicit check
            # for the goal state at the beginning is more efficient. The loop below
            # calculates cost for packages *not* at the goal.

            # Case: Package is on the ground (current_pos is a location string)
            # We check if current_pos is a known location by seeing if it's in our road graph keys.
            if current_pos in self.road_graph:
                current_location = current_pos
                # If the package is on the ground but not at the goal
                if current_location != goal_location:
                    total_pickup_cost += 1 # Needs pick-up
                    total_drop_cost += 1   # Needs drop at goal
                    # Needs transport from current_location to goal_location
                    drive_dist = self.shortest_paths.get((current_location, goal_location), float('inf'))
                    if drive_dist == float('inf'):
                         # Goal location unreachable from current location
                         return float('inf') # Problem is likely unsolvable from here
                    max_drive_cost = max(max_drive_cost, drive_dist)
                # else: package is at goal location on the ground, cost is 0 for this package, already handled by initial goal check

            # Case: Package is in a vehicle (current_pos is a vehicle string)
            elif package in packages_in_vehicles:
                vehicle = current_pos # current_pos is the vehicle name
                vehicle_location = vehicle_locations.get(vehicle) # Where is the vehicle?

                if vehicle_location is None:
                    # Vehicle location not found in state? Should not happen in a valid state.
                    # Indicates an unsolvable path or state error.
                    # print(f"Warning: Vehicle {vehicle} carrying {package} has no 'at' fact in state.")
                    return float('inf') # Indicate invalid or unsolvable state

                total_drop_cost += 1 # Always need to drop if in a vehicle

                # If the vehicle carrying the package is not at the goal location
                if vehicle_location != goal_location:
                    # Need to drive vehicle from its current location to goal_location
                    drive_dist = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))
                    if drive_dist == float('inf'):
                         # Goal location unreachable from vehicle's current location
                         return float('inf') # Problem is likely unsolvable from here
                    max_drive_cost = max(max_drive_cost, drive_dist)
                # else: vehicle is at goal location, only drop needed

            # Case: Package is not 'at' a location and not 'in' a vehicle.
            # This shouldn't happen in a valid PDDL state for a locatable object.
            # The initial check `if package not in current_locations` handles this by skipping.
            # If we didn't skip, we'd need to decide on a cost (e.g., infinity).

        # The total heuristic is the sum of individual pick-up and drop actions
        # plus the longest required drive distance.
        return total_pickup_cost + total_drop_cost + max_drive_cost

