from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Check if the number of parts matches the number of args for a basic sanity check
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It calculates the cost for each package
    independently, summing up the estimated actions (load, unload, and drive)
    needed for that package to reach its goal from its current state.

    # Assumptions
    - Each package needs to reach a specific goal location on the ground.
    - Vehicles have sufficient capacity for any package they might need to transport
      (capacity constraints are ignored).
    - Vehicles are always available when needed by a package (vehicle coordination
      is ignored).
    - Driving between connected locations costs 1 action.
    - Loading a package into a vehicle costs 1 action.
    - Unloading a package from a vehicle costs 1 action.
    - The road network is undirected (if road A B exists, road B A exists).
    - All locations relevant to package goals are connected in the road network.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representing the road network from static facts.
    - Precomputes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. For each package that has a goal location:
       a. Determine the package's current status: Is it on the ground at some location, or is it inside a vehicle?
       b. If the package is inside a vehicle, determine the vehicle's current location. This vehicle's location is the package's effective current location.
       c. Let `current_loc` be the package's effective current location and `goal_loc` be its goal location.
       d. If the package is on the ground at `goal_loc`, its contribution to the heuristic is 0.
       e. Otherwise, calculate the estimated cost for this package:
          - If the package is currently on the ground at `current_loc` (`current_loc != goal_loc`):
            Estimated cost = 1 (load) + shortest_distance(`current_loc`, `goal_loc`) (drive) + 1 (unload).
          - If the package is currently inside a vehicle which is at `current_loc`:
            Estimated cost = shortest_distance(`current_loc`, `goal_loc`) (drive) + 1 (unload).
       f. Add the estimated cost for this package to the total heuristic cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph for distance calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional

        self.locations = list(locations) # Store list of all locations

        # Precompute all-pairs shortest paths (distances) using BFS.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        in the road network using BFS. Assumes unit edge costs.
        Returns a dictionary {(loc1, loc2): distance}.
        """
        distances = {}
        # Assign a large value for unreachable pairs.
        # This value should be larger than any possible shortest path in a connected component.
        # Max possible path length is |locations| - 1.
        large_value = len(self.locations) + 1

        for start_node in self.locations:
            # Perform BFS starting from start_node
            queue = deque([(start_node, 0)])
            visited = {start_node: 0}
            distances[(start_node, start_node)] = 0 # Distance to self is 0

            while queue:
                current_loc, dist = queue.popleft()

                # Store distance if not already found or if a shorter path is found (BFS finds shortest for unit cost)
                # This check is actually redundant for BFS on unweighted graphs, but harmless.
                if (start_node, current_loc) not in distances or distances[(start_node, current_loc)] > dist:
                     distances[(start_node, current_loc)] = dist


                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                            distances[(start_node, neighbor)] = dist + 1 # Store distance immediately

        # Ensure all pairs have a distance (either computed or large_value)
        # This loop handles pairs where the target is unreachable from the source.
        for l1 in self.locations:
            for l2 in self.locations:
                 if (l1, l2) not in distances:
                     distances[(l1, l2)] = large_value

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # Map objects (packages, vehicles) to their current location or container.
        current_status = {} # {obj: location or vehicle}
        vehicle_locations = {} # {vehicle: location}

        # Populate current_status and vehicle_locations from the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                current_status[obj] = loc
                # A simple way to identify vehicles: check if the object name starts with 'v'
                # This assumes object naming conventions. A more robust way would be to parse object types from the domain,
                # but the instance files suggest this convention is used.
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1:]
                current_status[package] = vehicle # Package is inside vehicle

        total_cost = 0

        # Iterate through packages that have a goal location.
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground.
            # A package is at its goal if (at package goal_location) is true AND (in package vehicle) is false for any vehicle.
            # We can check this by looking at the current_status.
            # If current_status[package] is goal_location AND it's not a vehicle name, it's on the ground at the goal.
            pkg_current_status = current_status.get(package)

            if pkg_current_status is None:
                 # Package not found in state facts. This is unexpected in a valid state.
                 # Assign a large penalty.
                 # print(f"Warning: Package {package} not found in state {state}")
                 total_cost += len(self.locations) * 3 # Large penalty
                 continue # Move to the next package

            # Check if the package is currently inside a vehicle
            is_in_vehicle = pkg_current_status in vehicle_locations

            if not is_in_vehicle and pkg_current_status == goal_location:
                # Package is on the ground at its goal location
                continue # Cost is 0 for this package

            # Package is not at its goal on the ground. Calculate cost.
            current_loc = None # Effective current location of the package

            if is_in_vehicle:
                # Package is inside a vehicle. Effective location is the vehicle's location.
                vehicle = pkg_current_status
                current_loc = vehicle_locations.get(vehicle)
                if current_loc is None:
                     # Vehicle location not found? Unexpected.
                     # print(f"Warning: Vehicle {vehicle} carrying {package} not found at any location in state {state}")
                     total_cost += len(self.locations) * 3 # Large penalty
                     continue # Move to the next package

                # Cost: drive vehicle from current_loc to goal_location + unload
                # If current_loc == goal_location, drive_cost is 0.
                drive_cost = self.distances.get((current_loc, goal_location), len(self.locations) + 1)
                total_cost += drive_cost + 1 # 1 for unload action

            else:
                # Package is on the ground at pkg_current_status.
                current_loc = pkg_current_status

                # Cost: load + drive from current_loc to goal_location + unload
                # We already handled the case where current_loc == goal_location and on ground.
                # So if we are here, current_loc != goal_location.
                drive_cost = self.distances.get((current_loc, goal_location), len(self.locations) + 1)
                total_cost += 1 + drive_cost + 1 # 1 for load, 1 for unload

        return total_cost
