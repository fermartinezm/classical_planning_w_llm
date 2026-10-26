# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string or malformed inputs gracefully
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

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
# If running standalone for testing, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

from fnmatch import fnmatch
from collections import deque

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently, ignoring vehicle capacity constraints and
    potential conflicts or synergies between package movements.

    # Assumptions
    - The primary cost components are picking up a package, driving a vehicle,
      and dropping a package. Each is assumed to cost 1 action unit.
    - Vehicle capacity is assumed to be sufficient for any package movement.
    - Any location is reachable from any other location if a path exists in the
      road network. The shortest path distance is used for driving cost.
    - The heuristic does not consider vehicle availability or optimal vehicle
      assignment for packages on the ground. It assumes *a* vehicle can be used.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Identify all vehicle names from the initial state's capacity facts.
    - Build a graph representation of the road network from the static `road` facts.
    - Precompute the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. Identify the current location or container (vehicle) for every package
       that has a goal location defined, and the current location for every vehicle.
       This information is extracted by iterating through the facts in the current state.
    3. For each package `p` with a goal location `goal_l` (extracted during initialization):
       a. Check if the package `p` is already at its goal location `goal_l`
          (i.e., `(at p goal_l)` is true in the current state). If yes, the cost
          for this package is 0, proceed to the next package.
       b. If the package `p` is not at its goal location, determine its current status
          (on the ground at a location, or inside a vehicle) based on the information
          gathered in step 2.
          - If `p` is on the ground at a location `current_l` (where `current_l != goal_l`):
            - This package needs to be picked up (1 action).
            - A vehicle needs to drive from `current_l` to `goal_l`. The estimated
              cost for this is the shortest path distance `dist(current_l, goal_l)`.
            - The package needs to be dropped at `goal_l` (1 action).
            - Add `1 + drive_cost + 1` to the total cost.
          - If `p` is inside a vehicle `v`:
            - Find the current location `current_v_l` of vehicle `v` from the information
              gathered in step 2.
            - Vehicle `v` needs to drive from `current_v_l` to `goal_l`. The estimated
              cost for this is the shortest path distance `dist(current_v_l, goal_l)`.
            - The package needs to be dropped at `goal_l` (1 action).
            - Add `drive_cost + 1` to the total cost.
          - If the package's status is not found (neither `at` nor `in`), this indicates
            an issue with the state representation or problem definition. The heuristic
            assumes valid states where packages are always located or contained.
            In such cases, the heuristic skips this package, which is an optimistic
            assumption suitable for a non-admissible heuristic.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        location graph, and precomputing shortest path distances.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Identify all vehicle names from initial state facts (e.g., capacity).
        self.vehicles = set()
        for fact in self.task.initial_state:
             if match(fact, "capacity", "*", "*"):
                 _, vehicle_name, _ = get_parts(fact)
                 self.vehicles.add(vehicle_name)

        # Build the location graph from road facts.
        self.location_graph = {}
        all_locations_from_roads = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                all_locations_from_roads.add(l1)
                all_locations_from_roads.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on domain examples
                self.location_graph.setdefault(l2, set()).add(l1)

        # Ensure all locations mentioned in initial state or goals are included,
        # even if they have no roads (isolated nodes).
        all_relevant_locations = set()
        for fact in self.task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_relevant_locations.add(loc)
        for loc in self.goal_locations.values():
             all_relevant_locations.add(loc)

        # Add any relevant locations not found in road facts to the graph structure
        # as isolated nodes, so BFS can potentially start from them.
        for loc in all_relevant_locations:
             self.location_graph.setdefault(loc, set())

        # Precompute all-pairs shortest paths using BFS.
        self.distances = {}
        all_locations_in_graph = list(self.location_graph.keys())

        for start_node in all_locations_in_graph:
            queue = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0 # Distance to self is 0

            while queue:
                (current_node, dist) = queue.popleft()

                # Store distance from start_node to current_node
                self.distances[(start_node, current_node)] = dist

                # Explore neighbors
                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Define a large value to represent infinite distance for unreachable locations.
        # This is used if get_distance is called for locations not connected by roads.
        # In a solvable problem, relevant locations should be connected, but this adds robustness.
        # A distance larger than any possible path in a graph with N nodes is N.
        self._large_distance = len(all_locations_in_graph) + 1


    def get_distance(self, loc1, loc2):
         """Helper to get precomputed distance, returning a large value if path not found."""
         # Return precomputed distance, or large value if BFS didn't reach loc2 from loc1
         # or if loc1/loc2 were not part of the graph keys processed by BFS.
         return self.distances.get((loc1, loc2), self._large_distance)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # package -> location | vehicle
        current_package_status = {}
        # vehicle -> location
        current_vehicle_locations = {}

        # Populate current locations/status from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package on the ground
                     current_package_status[obj] = loc
                elif obj in self.vehicles: # It's a vehicle
                     current_vehicle_locations[obj] = loc
            elif predicate == "in":
                 package, vehicle = parts[1], parts[2]
                 # Ensure the object is a package we care about (has a goal)
                 if package in self.goal_locations:
                    current_package_status[package] = vehicle # Package is in a vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal
        for package, goal_location in self.goal_locations.items():
            # Check if the goal fact for this package is already true
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at its goal location on the ground

            # Package is not at its goal location on the ground.
            # Find its current status (location or vehicle).
            current_status = current_package_status.get(package)

            if current_status is None:
                 # This package is not found in any (at) or (in) facts in the state.
                 # This indicates an issue with the state representation or problem definition.
                 # For a non-admissible heuristic, we can skip this package.
                 continue

            is_in_vehicle = current_status in self.vehicles # Check if the status is a vehicle name

            if not is_in_vehicle:
                # Package is on the ground at current_status (which is a location)
                current_package_location = current_status
                # Cost: pick-up (1) + drive + drop (1)
                drive_cost = self.get_distance(current_package_location, goal_location)
                # If drive_cost is _large_distance, it means goal is unreachable from current location.
                # In a solvable problem, this shouldn't happen for relevant locations.
                # If it does, adding a large cost is appropriate.
                total_cost += 1 + drive_cost + 1
            else:
                # Package is in a vehicle (current_status is the vehicle name)
                vehicle = current_status
                # Find the vehicle's location
                current_vehicle_location = current_vehicle_locations.get(vehicle)

                if current_vehicle_location is None:
                     # Vehicle location unknown - problem with state or unreachable vehicle?
                     # Return a large value to penalize states where vehicle location is missing.
                     # This makes the state seem very far from the goal.
                     return self._large_distance * len(self.goal_locations) # Large cost

                # Cost: drive + drop (1)
                drive_cost = self.get_distance(current_vehicle_location, goal_location)
                # If drive_cost is _large_distance, it means goal is unreachable from vehicle's location.
                # Add a large cost.
                total_cost += drive_cost + 1

        return total_cost
