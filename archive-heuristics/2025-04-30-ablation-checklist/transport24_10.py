import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of arguments, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the cost of picking up, dropping off,
    and driving the necessary distances. It uses shortest path distances on the
    road network but simplifies by summing costs per package, ignoring vehicle
    capacity and potential for shared trips.

    # Assumptions
    - All road connections are bidirectional (even if only one direction is explicitly listed, the shortest path calculation assumes bidirectionality).
    - Vehicle capacity is not explicitly modeled in the heuristic cost, assuming any package can eventually be moved.
    - The cost of each action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Build a graph of locations based on `road` facts.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all packages and their goal locations from the task's goals.
    2.  For the current state, determine the current status of each package:
        - Is it at its goal location? If yes, it contributes 0 to the heuristic.
        - Is it at a non-goal location `l_current` (on the ground)?
        - Is it inside a vehicle `v`? If so, find the current location `l_v` of vehicle `v`.
    3.  For each package `p` that is *not* at its goal location `l_goal`:
        - Estimate the minimum actions needed *for this package* from its current state.
        - If `p` is `(at p l_current)` where `l_current != l_goal`:
            - It needs to be picked up (1 action).
            - It needs to be driven from `l_current` to `l_goal` (estimated by `shortest_path(l_current, l_goal)` drive actions).
            - It needs to be dropped (1 action).
            - Total estimated cost for this package: `1 (pick-up) + shortest_path(l_current, l_goal) + 1 (drop)`.
        - If `p` is `(in p v)` and vehicle `v` is `(at v l_v)`:
            - It needs to be driven from `l_v` to `l_goal` (estimated by `shortest_path(l_v, l_goal)` drive actions).
            - It needs to be dropped (1 action).
            - Total estimated cost for this package: `shortest_path(l_v, l_goal) + 1 (drop)`.
    4.  The total heuristic value is the sum of these estimated costs for all packages not yet at their goal. This relaxation ignores that multiple packages might share a vehicle trip, but provides a reasonable lower bound on actions involving the package itself and the necessary movement.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for shortest path calculations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the graph of locations from road facts.
        self.locations = set()
        self.graph = collections.defaultdict(set)
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph[l1].add(l2)
                self.graph[l2].add(l1) # Assume roads are bidirectional

        # Compute all-pairs shortest paths.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        Returns a dictionary distances[l1][l2] = shortest_path_cost.
        Unreachable locations will have distance float('inf').
        """
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """
        Performs Breadth-First Search from a start location to find distances
        to all other reachable locations.
        """
        dist = {loc: float('inf') for loc in self.locations}
        dist[start_loc] = 0
        queue = collections.deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.graph: # Ensure location exists in graph keys
                for neighbor in self.graph[current_loc]:
                    if dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[current_loc] + 1
                        queue.append(neighbor)
        return dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current status of packages and vehicles.
        # package_status: package_name -> ('at', location) or ('in', vehicle_name)
        # vehicle_location: vehicle_name -> location
        package_status = {}
        vehicle_location = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj_name, loc_name = parts[1], parts[2]
                # Need to distinguish between vehicles and packages
                # We don't have type info here, rely on object naming conventions or check against known objects
                # A robust way would be to get types from the task, but for a heuristic,
                # we can assume 'v' prefix for vehicles and 'p' for packages based on examples.
                if obj_name.startswith('v'): # Assuming vehicles start with 'v'
                     vehicle_location[obj_name] = loc_name
                elif obj_name.startswith('p'): # Assuming packages start with 'p'
                     package_status[obj_name] = ('at', loc_name)
            elif parts[0] == "in":
                package_name, vehicle_name = parts[1], parts[2]
                package_status[package_name] = ('in', vehicle_name)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location defined.
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state at all, it's likely an error or not relevant
            if package not in package_status:
                 continue # Or handle as an unreachable goal? For heuristic, assume reachable.

            current_status, current_loc_or_vehicle = package_status[package]

            # Check if the package is already at its goal location
            if current_status == 'at' and current_loc_or_vehicle == goal_location:
                continue # Package is already at the goal, cost is 0 for this package.

            # Package is not at its goal. Estimate cost to get it there.
            if current_status == 'at':
                # Package is on the ground at current_loc_or_vehicle
                current_location = current_loc_or_vehicle
                # Cost includes pick-up, drive, and drop
                drive_cost = self.distances[current_location].get(goal_location, float('inf'))
                total_cost += 1 # pick-up
                total_cost += drive_cost
                total_cost += 1 # drop
            elif current_status == 'in':
                # Package is inside a vehicle. Find vehicle's location.
                vehicle_name = current_loc_or_vehicle
                # If vehicle location is unknown (e.g., vehicle not in state facts - unlikely in valid states),
                # treat as unreachable or high cost.
                if vehicle_name not in vehicle_location:
                     # This state might be invalid or represent a complex scenario not covered.
                     # Assign a very high cost.
                     return float('inf')

                vehicle_current_location = vehicle_location[vehicle_name]
                # Cost includes drive (with package) and drop
                drive_cost = self.distances[vehicle_current_location].get(goal_location, float('inf'))
                total_cost += drive_cost
                total_cost += 1 # drop

            # If any required location is unreachable, the total cost will be infinity.
            if total_cost == float('inf'):
                 return float('inf')


        return total_cost

