import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (copied from Logistics example, they are general)
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A more robust check could be added, but for typical PDDL facts and patterns,
    # checking length and then zipping with fnmatch is usually sufficient.
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move
    all packages to their goal locations. It sums the estimated cost for
    each package independently, ignoring vehicle capacity and shared travel.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - The shortest path distance between locations is the minimum number of drive actions.
    - Vehicle capacity constraints are ignored (relaxation).
    - A vehicle is always available to pick up a package or drop one off (relaxation).

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds a graph of locations based on `road` facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not at its goal location:
    1. Determine the package's current status: Is it on the ground at a location `l_current`, or is it inside a vehicle `v` which is at location `l_v`?
    2. Determine the package's goal location `l_goal`.
    3. Estimate the minimum actions required for *this package* to reach its goal:
       - If the package is on the ground at `l_current` (`l_current != l_goal`):
         It needs to be picked up (1 action), the vehicle needs to drive from `l_current` to `l_goal` (`dist(l_current, l_goal)` drive actions), and it needs to be dropped (1 action).
         Estimated cost for this package: `1 + dist(l_current, l_goal) + 1`.
       - If the package is inside a vehicle `v` which is at `l_v`:
         - If `l_v != l_goal`: The vehicle needs to drive from `l_v` to `l_goal` (`dist(l_v, l_goal)` drive actions), and the package needs to be dropped (1 action).
           Estimated cost for this package: `dist(l_v, l_goal) + 1`.
         - If `l_v == l_goal`: The package just needs to be dropped (1 action).
           Estimated cost for this package: `1`.
    4. The total heuristic value is the sum of these estimated costs for all packages not yet at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the location graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Extract all locations and build the road graph
        locations = set()
        road_graph = collections.defaultdict(list)

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                road_graph[loc1].append(loc2)
            # Also collect locations from capacity-predecessor facts if any objects
            # of type location appear there, though unlikely based on domain.
            # Collect locations from initial state 'at' predicates later.

        # Collect locations mentioned in initial state 'at' predicates
        # We don't have direct access to initial_state here, but we can assume
        # the Task object provides all relevant static facts and goals.
        # Let's iterate through all facts in static and goals to find locations.
        # A more robust way would be to parse the problem file objects section,
        # but we are limited to the Task object structure.
        # Let's refine location extraction by looking at all facts in static and goals.
        # A location object will appear as the second argument of 'at' or arguments of 'road'.
        all_facts_in_task = set(static_facts) | set(task.goals) # Initial state facts are not directly available in task.static
        # A better approach is to assume locations are objects of type 'location'
        # and they appear in 'road' facts. Let's stick to roads for graph building.
        # We will collect all locations mentioned in road facts.

        self.locations = list(locations) # Convert to list for consistent indexing if needed, though dict is better

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node, road_graph)

        # 3. Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            # Assuming goal facts are primarily (at package location)
            if match(goal, "at", "*", "*"):
                predicate, obj, location = get_parts(goal)
                # Need to check if obj is a package. The Task object doesn't give types.
                # We assume any object with an 'at' goal is a package for this domain.
                self.goal_locations[obj] = location
            # Ignore other potential goal predicates if any

    def _bfs(self, start_node, graph):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Check if node has outgoing roads
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        # We need to iterate through the state to find the current status of
        # packages and vehicles.
        current_package_status = {} # Maps package -> ('at', loc) or ('in', veh)
        vehicle_locations = {}      # Maps vehicle -> loc

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, location = parts[1], parts[2]
                # Need to distinguish packages from vehicles.
                # A simple way is to check if the object is in our goal_locations keys (packages)
                # or if it's not in goal_locations (likely a vehicle or other object).
                # This is a heuristic assumption based on typical problem structure.
                if obj in self.goal_locations: # Assume it's a package if it has a goal location
                     current_package_status[obj] = ('at', location)
                else: # Assume it's a vehicle or other locatable
                     # Check if it's a vehicle by looking for capacity facts or by name pattern if available
                     # For simplicity, let's assume anything with an 'at' predicate not in goal_locations is a vehicle
                     # This might be fragile if other locatables exist, but fits the domain types.
                     vehicle_locations[obj] = location
            elif parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 # Assume the first argument of 'in' is always a package
                 current_package_status[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if ('at', goal_location) in current_package_status.get(package, (None, None)):
                 continue # Package is already at its goal, cost is 0 for this package

            # Find the package's current status
            status, loc_or_veh = current_package_status.get(package, (None, None))

            if status == 'at':
                # Package is on the ground at loc_or_veh (current_location)
                current_location = loc_or_veh
                # Cost = pick-up (1) + drive (dist) + drop (1)
                drive_cost = self.distances[current_location].get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # If goal is unreachable, return infinity (or a very large number)
                    return float('inf')
                total_cost += 1 + drive_cost + 1 # pick + drive + drop

            elif status == 'in':
                # Package is inside vehicle loc_or_veh (vehicle)
                vehicle = loc_or_veh
                # Find the vehicle's current location
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                    # This shouldn't happen in a valid state, but handle defensively
                    # If vehicle location is unknown, package is stuck. Infinite cost.
                     return float('inf')

                # Cost = drive (dist) + drop (1)
                drive_cost = self.distances[vehicle_location].get(goal_location, float('inf'))
                if drive_cost == float('inf'):
                    # If goal is unreachable, return infinity
                    return float('inf')
                total_cost += drive_cost + 1 # drive + drop

            # If status is None, the package is not mentioned in 'at' or 'in' facts,
            # which implies an invalid state or an unhandled initial state representation.
            # Assuming valid states always have packages either 'at' a location or 'in' a vehicle.
            # If a package is not found in current_package_status but has a goal, it's misplaced.
            # This case is covered by the .get(package, (None, None)) returning None.
            # However, the loop only iterates over packages *with goals*. If a package
            # with a goal isn't in the state facts at all, something is wrong with the state representation.
            # Assuming state contains facts for all objects.

        return total_cost

# Example Usage (assuming you have a Task object named 'task'):
# heuristic = transportHeuristic(task)
# h_value = heuristic(current_node) # where current_node has a 'state' attribute
