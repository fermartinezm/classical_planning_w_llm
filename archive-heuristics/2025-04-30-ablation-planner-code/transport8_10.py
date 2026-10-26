# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

import collections
from fnmatch import fnmatch

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to check if a fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic: # Inherit from Heuristic in the actual environment
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each misplaced package to its goal location, assuming vehicles are
    always available where needed and ignoring vehicle capacity constraints.
    It sums the individual costs for each package.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles can carry packages.
    - Roads are bidirectional if defined both ways.
    - Vehicle capacity is ignored.
    - Vehicle availability at pickup locations is assumed (no cost to move a vehicle to a package).

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goal conditions.
    - Builds a graph of locations based on static `road` facts.
    - Precomputes shortest path distances (number of `drive` actions) between all pairs of locations using BFS.
    - Identifies objects by their types (packages, vehicles, locations) from the task's object definitions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  Initialize the total heuristic cost `h` to 0.
    2.  Identify the current status (location or vehicle containment) for all packages and vehicles in the current state.
        -   Create mappings: `package_location` (package on ground -> location), `package_in_vehicle` (package -> vehicle), `vehicle_location` (vehicle -> location).
    3.  For each package `p` that has a goal location `goal_l` (extracted during initialization):
        -   Check if the goal `(at p goal_l)` is already satisfied in the current state. If it is, the cost for this package is 0, and we proceed to the next package.
        -   If the goal is not satisfied, determine the package's current status:
            -   If `p` is on the ground at `current_l` (i.e., `(at p current_l)` is in the state):
                -   The package needs to be picked up (1 action), transported from `current_l` to `goal_l` (minimum `distance(current_l, goal_l)` drive actions), and dropped at `goal_l` (1 action).
                -   Add `distance(current_l, goal_l) + 2` to `h`. (Assumes a vehicle is available at `current_l`).
            -   If `p` is inside a vehicle `v` (i.e., `(in p v)` is in the state):
                -   Find the location `vehicle_l` of vehicle `v` (i.e., `(at v vehicle_l)` is in the state).
                -   If `vehicle_l` is not the `goal_l`: The package needs to be transported from `vehicle_l` to `goal_l` (minimum `distance(vehicle_l, goal_l)` drive actions) and dropped at `goal_l` (1 action).
                -   Add `distance(vehicle_l, goal_l) + 1` to `h`.
                -   If `vehicle_l` is the `goal_l`: The package only needs to be dropped at `goal_l` (1 action).
                -   Add `1` to `h`.
    4.  Return the total accumulated cost `h`.

    The distance calculation uses precomputed shortest paths on the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        object types, and precomputing distances between locations.
        """
        # Inherit goals and static facts from the base task object
        self.goals = task.goals
        static_facts = task.static
        objects_info = task.objects # Assuming task provides object names and types

        # Identify objects by type
        self.packages = {obj for obj, obj_type in objects_info['types'].items() if obj_type == 'package'}
        self.vehicles = {obj for obj, obj_type in objects_info['types'].items() if obj_type == 'vehicle'}
        self.locations = {obj for obj, obj_type in objects_info['types'].items() if obj_type == 'location'}

        # Store goal locations for each package
        self.package_goals = {}
        # Assuming task.goals is a frozenset of individual goal facts like '(at p1 l2)'
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            # Ensure the goal is an 'at' predicate for a package
            if parts[0] == "at" and len(parts) == 3:
                 package, location = parts[1], parts[2]
                 # Verify types if possible, but relying on task.objects might be better
                 # For now, assume any 'at' goal involving a package is relevant
                 if package in self.packages and location in self.locations:
                    self.package_goals[package] = location


        # Build the road network graph and precompute distances
        self.distances = self._precompute_distances(static_facts)

    def _precompute_distances(self, static_facts):
        """
        Builds the road graph and computes shortest path distances between
        all pairs of locations using BFS.
        """
        graph = collections.defaultdict(set)
        all_locations_in_roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                graph[loc1].add(loc2)
                graph[loc2].add(loc1) # Assuming bidirectionality based on example
                all_locations_in_roads.add(loc1)
                all_locations_in_roads.add(loc2)

        # Ensure all locations defined in objects are considered, even if isolated
        all_locations_from_objects = self.locations
        # Combine locations from objects and roads
        all_relevant_locations = all_locations_from_objects.union(all_locations_in_roads)


        distances = {}
        # Use all relevant locations as potential start nodes for BFS
        for start_node in all_relevant_locations:
            # BFS to find distances from start_node to all other nodes
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            distances[(start_node, start_node)] = 0

            while queue:
                current_node, dist = queue.popleft()

                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Assign a large distance for unreachable pairs
        # Use a value larger than any possible path in a connected component
        # If there are N locations, max path length is N-1. N is len(all_relevant_locations).
        large_distance = len(all_relevant_locations) + 1
        for l1 in all_relevant_locations:
            for l2 in all_relevant_locations:
                if (l1, l2) not in distances:
                     distances[(l1, l2)] = large_distance # Represents unreachable

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state facts (frozenset of strings)

        # Map current locations/status for packages and vehicles
        package_location = {}     # package -> location (if on ground)
        package_in_vehicle = {} # package -> vehicle (if in vehicle)
        vehicle_location = {}     # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Check if the object and location are of the correct types
                if obj in self.packages and loc in self.locations:
                    package_location[obj] = loc
                elif obj in self.vehicles and loc in self.locations:
                    vehicle_location[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                # Check if the objects are of the correct types
                if pkg in self.packages and veh in self.vehicles:
                     package_in_vehicle[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # Check if the state is a goal state. If yes, heuristic is 0.
        # This is important for the heuristic to be 0 ONLY at goal states.
        # A state is a goal state if all goal facts are present.
        is_goal_state = all(goal in state for goal in self.goals)
        if is_goal_state:
             return 0

        # If not a goal state, calculate the heuristic
        for package, goal_location in self.package_goals.items():
            # Check if the goal for this specific package is met (at the goal location on the ground)
            if (f"(at {package} {goal_location})") in state:
                continue # This package is already at its goal location on the ground

            # Find package's current status
            if package in package_location:
                # Package is on the ground at package_location[package]
                current_l = package_location[package]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                # Assumes a vehicle is available at current_l
                dist = self.distances.get((current_l, goal_location), len(self.locations) + 1) # Use large distance if unreachable
                total_cost += dist + 2

            elif package in package_in_vehicle:
                # Package is inside vehicle package_in_vehicle[package]
                vehicle = package_in_vehicle[package]
                # Find vehicle's location
                if vehicle in vehicle_location:
                    vehicle_l = vehicle_location[vehicle]
                    # Cost: drive (distance) + drop (1)
                    dist = self.distances.get((vehicle_l, goal_location), len(self.locations) + 1) # Use large distance if unreachable
                    total_cost += dist + 1
                else:
                    # This case indicates a package is in a vehicle, but the vehicle's location is unknown
                    # (e.g., not in the state facts). This shouldn't happen in valid states,
                    # but assign a high cost defensively.
                    # Use a cost that makes this state highly undesirable.
                    total_cost += len(self.locations) * 2 + 3 # Arbitrarily large cost

            else:
                 # Package is not on the ground and not in a vehicle.
                 # This indicates an invalid state representation or a package that
                 # has been 'lost'. Assign a very high cost.
                 # Use a cost that makes this state highly undesirable.
                 total_cost += len(self.locations) * 3 + 5 # Arbitrarily very large cost


        return total_cost
