from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is provided as per problem description
# from heuristics.heuristic_base import Heuristic

# Note: The problem description implies a Heuristic base class exists.
# If running this code standalone without that base class, you would need
# to define a dummy one or provide the actual base class implementation.
# For the purpose of providing the solution as requested, we assume the
# base class 'Heuristic' is available in the environment where this code
# will be used, and we inherit from it.

# Helper functions used by the heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The heuristic class definition
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing, for each
    misplaced package, the estimated minimum actions required to get it to its
    goal location. The estimate for a single package includes costs for pick-up,
    transport (shortest path distance), and drop-off.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity constraints are ignored for the heuristic calculation.
    - Vehicle availability at a package's location is ignored; it's assumed a vehicle
      will eventually reach the package.
    - The heuristic sums costs per package, ignoring potential optimizations from
      transporting multiple packages in one vehicle trip. This makes it potentially
      inadmissible but can still be effective for greedy search.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Build the road network graph from the static `road` predicates.
    - Compute the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. For each package `p` that has a goal location `goal_l`:
       a. Check if `p` is already at `goal_l` in the current state. If yes, the cost for this package is 0, continue to the next package.
       b. Find the current status of package `p`:
          - Is it at a location `current_l` (on the ground)? (i.e., `(at p current_l)` is in the state)
          - Is it inside a vehicle `v`? (i.e., `(in p v)` is in the state)
       c. If `p` is at `current_l` (on the ground):
          - The estimated cost for this package is 1 (pick-up) + shortest_distance(`current_l`, `goal_l`) (drive) + 1 (drop). Add this to the total cost.
       d. If `p` is inside vehicle `v`:
          - Find the current location of vehicle `v`, say `current_v_l`. (i.e., `(at v current_v_l)` is in the state)
          - The estimated cost for this package is shortest_distance(`current_v_l`, `goal_l`) (drive) + 1 (drop). Add this to the total cost.
    3. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        all-pairs shortest paths in the road network.
        """
        # Call the base class constructor to store goals and static facts
        super().__init__(task)

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            if match(goal, "at", "*", "*"):
                predicate, package, location = get_parts(goal)
                self.package_goals[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()

        # Collect all locations mentioned in road facts
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                predicate, l1, l2 = get_parts(fact)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph: # Ensure l2 is also a key
                     self.road_graph[l2] = []
                self.road_graph[l1].append(l2)
                locations.add(l1)
                locations.add(l2)

        # Collect all locations mentioned in initial state (at facts)
        # We need the initial state to find all possible locations in the problem
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 predicate, obj, loc = get_parts(fact)
                 if loc not in self.road_graph:
                     self.road_graph[loc] = []
                 locations.add(loc)

        # Collect all locations mentioned in goal state (at facts)
        for goal_loc in self.package_goals.values():
             if goal_loc not in self.road_graph:
                 self.road_graph[goal_loc] = []
             locations.add(goal_loc)

        # Compute all-pairs shortest paths.
        self.distances = {}
        for start_node in locations:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            # Get neighbors, handle locations that might be in init/goals but not in road facts
            # Ensure current_node is a key in road_graph, even if it has no outgoing roads
            neighbors = self.road_graph.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    def get_distance(self, l1, l2):
        """Returns the shortest distance between two locations, or infinity if unreachable."""
        # If locations are the same, distance is 0.
        if l1 == l2:
            return 0
        # Look up pre-calculated distance. If not found, they are unreachable.
        # The BFS ensures all locations from init/goals/roads are start nodes,
        # so if l1 is a known location, the distance to any reachable l2 will be in distances.
        # If l1 itself is not a known location (which shouldn't happen in valid states),
        # or if l2 is unreachable from l1, get() will return the default (infinity).
        return self.distances.get((l1, l2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Heuristic is 0 if all goals are met
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        current_locations = {} # Maps locatable object (package or vehicle) to its location
        package_in_vehicle = {} # Maps package to the vehicle it's in

        for fact in state:
            if match(fact, "at", "*", "*"):
                predicate, obj, loc = get_parts(fact)
                current_locations[obj] = loc
            elif match(fact, "in", "*", "*"):
                predicate, package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that is not at its goal.
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location
            # Check for the exact fact string in the state
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Find the current status of the package
            current_package_effective_location = None # Location if on ground or vehicle's location if in vehicle
            is_in_vehicle = False

            if package in package_in_vehicle:
                 is_in_vehicle = True
                 vehicle_carrying_package = package_in_vehicle[package]
                 # The package is in a vehicle, its effective location is the vehicle's location
                 current_vehicle_location = current_locations.get(vehicle_carrying_package)
                 if current_vehicle_location is None:
                     # Vehicle carrying the package is not located anywhere. Unsolvable.
                     return float('inf')
                 current_package_effective_location = current_vehicle_location # Effective location for distance calculation
            else: # Package is not in a vehicle, it must be at a location on the ground
                 current_package_effective_location = current_locations.get(package)
                 if current_package_effective_location is None:
                     # Package is not 'at' anywhere and not 'in' a vehicle. Unsolvable.
                     return float('inf')

            # Calculate cost based on package status
            if is_in_vehicle:
                # Package is in a vehicle at current_package_effective_location (which is vehicle's location)
                # Needs vehicle to drive from current_package_effective_location to goal_location, then drop
                drive_cost = self.get_distance(current_package_effective_location, goal_location)
                if drive_cost == float('inf'):
                    return float('inf') # Goal location unreachable from vehicle location
                total_cost += drive_cost + 1 # +1 for drop action
            else:
                # Package is on the ground at current_package_effective_location
                # Needs pick-up, vehicle drive from current_package_effective_location to goal_location, then drop
                drive_cost = self.get_distance(current_package_effective_location, goal_location)
                if drive_cost == float('inf'):
                    return float('inf') # Goal location unreachable from package location
                total_cost += 1 + drive_cost + 1 # +1 for pick-up, +1 for drop

        return total_cost
