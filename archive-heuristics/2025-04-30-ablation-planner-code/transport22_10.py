import collections
from fnmatch import fnmatch
# Assuming Heuristic base class is available in the execution environment
from heuristics.heuristic_base import Heuristic

# Utility function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Utility function to check if a PDDL fact matches a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing these individual estimates.
    It considers the actions needed to pick up, drive, and drop the package.
    The cost of driving is estimated by the shortest path distance in the road network.

    # Assumptions
    - The road network is undirected (if road l1 l2 exists, road l2 l1 exists).
    - Vehicle capacity is not explicitly modeled in the cost calculation for individual packages. It's assumed a vehicle with sufficient capacity will eventually be available.
    - Conflicts over vehicles or locations are ignored.
    - The cost of moving a vehicle to a package's location for pickup is not explicitly added if the package is on the ground; it's implicitly part of the 'pick-up' step cost estimate which assumes a vehicle is or can get there.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Parses object definitions to map object names to their types (e.g., package, vehicle, location).
    - Builds the road network graph from static `(road l1 l2)` facts.
    - Identifies all location objects declared in the problem.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a goal location specified and is not yet satisfying that goal:
    1. Determine the package's current status: Is it on the ground at some location, or inside a vehicle?
    2. If the package is on the ground at `loc_current` (and `loc_current` is not the goal):
       - Estimate the cost as 1 (pick-up) + `dist(loc_current, loc_goal)` (drive) + 1 (drop).
       - Total cost for this package: `2 + dist(loc_current, loc_goal)`.
       - If `loc_current` and `loc_goal` are the same, the package is on the ground at the goal, but the goal fact `(at p l)` might not be in the state yet if it was just dropped. However, the check `f"(at {package} {goal_location})" in state` handles the true goal state. If it's at the location but not the goal state (e.g., just dropped), the cost is 0 for this package.
    3. If the package is inside a vehicle `v`:
       - Find the current location of vehicle `v`, say `loc_v`.
       - Estimate the cost as `dist(loc_v, loc_goal)` (drive) + 1 (drop).
       - Total cost for this package: `1 + dist(loc_v, loc_goal)`.
    4. If the package is already satisfying its goal condition `(at package goal_location)` in the state, the cost for this package is 0.
    5. The total heuristic value is the sum of the estimated costs for all packages. If any required location is unreachable, the heuristic returns infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        # Assuming task.goals is an iterable of goal fact strings like `'(at p1 l2)'`
        self.goals = task.goals
        self.static = task.static
        # Assuming task.objects is a list of strings like 'v1 - vehicle', 'l1 - location'
        self.objects = task.objects

        # Parse objects to map object name to type
        self.object_types = {}
        if hasattr(task, 'objects') and isinstance(task.objects, list):
             for obj_str in task.objects:
                 parts = obj_str.split(' - ')
                 if len(parts) == 2:
                     self.object_types[parts[0]] = parts[1]

        # Extract goal locations for packages
        self.package_goals = {}
        for goal_fact_str in self.goals:
             if match(goal_fact_str, "at", "*", "*"):
                 obj, loc = get_parts(goal_fact_str)[1:]
                 # Check if the object is a package using the parsed object types
                 if self.object_types.get(obj) == 'package':
                     self.package_goals[obj] = loc

        # Build the road network graph and identify locations
        self.locations = set()
        self.road_graph = collections.defaultdict(set)
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assuming roads are bidirectional
                self.locations.add(l1)
                self.locations.add(l2)

        # Add all declared locations from objects list, even if they have no roads
        for obj_name, obj_type in self.object_types.items():
            if obj_type == 'location':
                self.locations.add(obj_name)

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in self.locations:
             # Cannot start BFS from a node not in our location set
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_loc = queue.popleft()

            # Get neighbors from the road graph
            # Check if current_loc is in the graph keys before accessing
            neighbors = self.road_graph.get(current_loc, set())

            for neighbor in neighbors:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 not in self.locations or loc2 not in self.locations:
             # If either location is not a known location object, they are unreachable
             return float('inf')
        # Use .get for safety, although BFS should populate all reachable distances
        return self.distances[loc1].get(loc2, float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Track current location/status of all locatables (packages and vehicles)
        current_status = {} # Maps object name to its location or vehicle it's in
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                current_status[obj] = loc
            elif match(fact, "in", "*", "*"):
                 pkg, veh = get_parts(fact)[1:]
                 current_status[pkg] = veh # Store the vehicle name

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if package is in the current state (it should be if problem is well-formed)
            if package not in current_status:
                 # This package is not mentioned in 'at' or 'in' facts.
                 # If a package is missing from the state, it cannot reach its goal.
                 return float('inf')

            current_whereabouts = current_status[package]

            # Case 1: Package is already at its goal location on the ground
            # Check if the goal fact is present in the state
            if f"(at {package} {goal_location})" in state:
                continue # Package is at goal, cost is 0 for this package

            # Case 2: Package is on the ground at a different location
            # Check if the current_whereabouts is a location (not a vehicle name)
            elif current_whereabouts in self.locations:
                 loc_current = current_whereabouts
                 # Package is at loc_current, needs to go to goal_location
                 # Actions: pick-up (1) + drive (dist) + drop (1)
                 drive_cost = self.get_distance(loc_current, goal_location)
                 if drive_cost == float('inf'):
                     # Cannot reach goal location from current location
                     return float('inf') # Problem is likely unsolvable from this state

                 total_cost += 1 # pick-up
                 total_cost += drive_cost # drive
                 total_cost += 1 # drop

            # Case 3: Package is inside a vehicle
            else: # current_whereabouts is a vehicle name
                 vehicle = current_whereabouts
                 # Find the location of the vehicle
                 if vehicle not in current_status or current_status[vehicle] not in self.locations:
                      # Vehicle location unknown or not a valid location.
                      return float('inf') # Problem is likely unsolvable from this state

                 loc_v = current_status[vehicle]

                 # Vehicle is at loc_v, needs to drive to goal_location and drop package
                 # Actions: drive (dist) + drop (1)
                 drive_cost = self.get_distance(loc_v, goal_location)
                 if drive_cost == float('inf'):
                     # Cannot reach goal location from vehicle's current location
                     return float('inf') # Problem is likely unsolvable from this state

                 total_cost += drive_cost # drive
                 total_cost += 1 # drop

        # If we iterated through all packages and none added infinity, return the sum.
        # If total_cost is 0, it means all packages were already satisfying their goal condition.
        return total_cost

