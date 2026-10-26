# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic
from collections import deque
import math

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS helper functions for shortest path calculation
def build_graph(locations, road_facts):
    """Builds a graph from locations and road facts."""
    graph = {loc: set() for loc in locations}
    for fact in road_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'road':
            l1, l2 = parts[1], parts[2]
            # Only add roads between locations that are in our collected set
            if l1 in graph and l2 in graph:
                graph[l1].add(l2)
                graph[l2].add(l1) # Assuming bidirectional roads
    return graph

def bfs(graph, start_node):
    """Performs BFS to find shortest distances from a start node."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., isolated location not in collected set)
         # This case should ideally not happen if all_locations is collected correctly.
         # If it does, we can't reach anything from here except maybe itself (dist 0).
         # Returning distances with inf for others is appropriate.
         # Check if start_node is even a known location before returning empty distances
         if start_node in distances: # Should be true if start_node was in the initial locations list
             distances[start_node] = 0
         return distances # Return distances with inf for others

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        # Iterate through neighbors using .get() for safety, though graph should be complete
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
    return distances

def collect_all_locations(task):
    """Collects all unique locations mentioned in the task."""
    locations = set()
    # From static facts (roads)
    for fact in task.static:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'road':
            locations.add(parts[1])
            locations.add(parts[2])
    # From initial state (at)
    for fact in task.initial_state:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'at':
             locations.add(parts[2])
    # From goals (at)
    for fact in task.goals:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] == 'at':
             locations.add(parts[2])
    return list(locations) # Return as list

def compute_all_pairs_shortest_paths(task):
    """Computes shortest path distances between all pairs of relevant locations."""
    all_locations = collect_all_locations(task)
    road_facts = [fact for fact in task.static if get_parts(fact)[0] == 'road']
    graph = build_graph(all_locations, road_facts) # Build graph including isolated locations

    all_distances = {}
    for start_loc in all_locations:
        all_distances[start_loc] = bfs(graph, start_loc)
    return all_distances

# Heuristic class definition
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location independently. It sums the minimum actions needed for
    each package, considering whether it's currently on the ground or inside a vehicle,
    and the shortest path distance to its goal.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Roads are bidirectional.
    - The heuristic ignores vehicle capacity constraints.
    - The heuristic ignores the cost of moving a vehicle to a package's location if the package is on the ground and no vehicle is present. It only counts the drive cost for the package itself once it's assumed to be in a vehicle.
    - The shortest path distance between locations is the minimum number of drive actions.
    - The heuristic assumes all locations mentioned in the problem (init, goal, roads) are part of the relevant graph.
    - Packages are identified by being the first argument of an 'at' predicate in the goal state. Vehicles are identified as objects in 'at' predicates in the initial state that are not packages.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Collects all relevant locations from initial state, goal state, and static road facts.
    - Builds a graph of locations based on static road facts, including isolated locations.
    - Computes all-pairs shortest path distances between all relevant locations using BFS.
    - Identifies package and vehicle names based on goal and initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. In the constructor (`__init__`):
       - Store package goal locations by parsing the task's goal facts (specifically 'at' predicates). Identify package names from these goals.
       - Identify vehicle names from initial state 'at' facts (objects that are not packages).
       - Identify all unique locations present in the initial state ('at' predicates), goal state ('at' predicates), and static road facts ('road' predicates).
       - Construct a graph where nodes are these locations and edges represent bidirectional roads from static facts. Include isolated locations as nodes.
       - Compute and store the shortest path distance between every pair of locations using BFS.

    2. In the heuristic function (`__call__`):
       - Check if the current state is a goal state. If yes, return 0.
       - Initialize the total heuristic cost to 0.
       - Determine the current state of all packages and vehicles by iterating through the state facts:
         - Map packages (identified in init/goals) to their ground location if `(at p l)` is true.
         - Map vehicles (identified in init) to their location if `(at v l)` is true.
         - Keep track of which packages are inside which vehicles if `(in p v)` is true.
       - For each package whose goal location is known (from `self.package_goals`):
         - Check if the goal fact `(at package goal_location)` is already true in the current state. If yes, this package is done, add 0 cost for it.
         - If the package is not at its goal location:
           - Determine the package's effective current location:
             - If the package is currently inside a vehicle, its effective location is the vehicle's current location.
             - If the package is on the ground, its effective location is its ground location.
           - If the package's location is unknown (e.g., not in state facts), return infinity as the state is likely invalid or unsolvable.
           - Find the shortest path distance from the effective current location to the package's goal location using the precomputed distances. If the goal is unreachable from this location, the distance is infinite.
           - If the distance is infinite, the state is likely unsolvable, return infinity immediately.
           - Calculate the minimum actions required for this package:
             - If the package is currently on the ground: 1 (pick-up) + distance + 1 (drop).
             - If the package is currently inside a vehicle: 1 (drop) + distance.
           - Add this cost to the total heuristic cost.
       - Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Extract package goals and identify package names
        self.package_goals = {}
        package_names = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.package_goals[package] = location
                package_names.add(package)
        self.package_names = list(package_names) # Store list of package names

        # 2. Identify vehicle names from initial state (objects in 'at' that are not packages)
        self.vehicle_names = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                 obj = parts[1]
                 if obj not in self.package_names:
                     self.vehicle_names.add(obj)
        self.vehicle_names = list(self.vehicle_names)

        # 3. Collect all relevant locations and compute distances
        self.distances = compute_all_pairs_shortest_paths(task)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
            return 0

        total_cost = 0
        pkg_ground_loc = {} # package -> ground location
        pkg_in_veh = {} # package -> vehicle
        veh_current_loc = {} # vehicle -> location

        # Populate current state mappings
        for fact in state:
            parts = get_parts(fact)
            if len(parts) >= 2:
                predicate = parts[0]
                obj = parts[1]

                if predicate == 'at' and len(parts) == 3:
                    loc = parts[2]
                    if obj in self.package_names:
                         pkg_ground_loc[obj] = loc
                    elif obj in self.vehicle_names:
                         veh_current_loc[obj] = loc
                    # Ignore other 'at' facts if any (e.g., 'at-robby' in gripper example)
                elif predicate == 'in' and len(parts) == 3:
                    pkg, veh = obj, parts[2]
                    # Ensure the objects are of the expected types (package and vehicle)
                    if pkg in self.package_names and veh in self.vehicle_names:
                        pkg_in_veh[pkg] = veh
                    # Ignore other 'in' facts if any


        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # Check if the goal fact (at package goal_location) is already true
            # This handles packages already at their final destination on the ground.
            if f"(at {package} {goal_location})" in state:
                continue # Goal met for this package

            # Goal not met. Calculate cost for this package.
            l_eff_curr = None
            if package in pkg_in_veh:
                veh = pkg_in_veh[package]
                # Get vehicle's location. If vehicle location is unknown, the state is problematic.
                l_eff_curr = veh_current_loc.get(veh)
            elif package in pkg_ground_loc:
                l_eff_curr = pkg_ground_loc[package] # Get package's ground location
            # else: package exists in goals but is not 'at' any location and not 'in' any vehicle? (invalid state assumption)

            if l_eff_curr is None:
                 # Package location is unknown - potentially an invalid state or unhandled case
                 # Return infinity to indicate a likely dead end or unsolvable state
                 return float('inf')

            # Get shortest path distance from effective current location to goal location
            # Use .get() with default float('inf') to handle cases where locations might not be in the distance map
            # (e.g., isolated locations not connected by roads, or goal location unreachable)
            # We need to check if l_eff_curr is a valid key in self.distances first.
            if l_eff_curr not in self.distances:
                 # Effective current location is not a known location in the graph
                 return float('inf') # Unreachable

            drive_cost = self.distances[l_eff_curr].get(goal_location, float('inf'))

            if drive_cost == float('inf'):
                 # Goal location is unreachable from the package's current effective location
                 # This state is likely unsolvable
                 return float('inf')

            # Determine base cost (pickup/drop)
            if package in pkg_in_veh:
                # Package is in a vehicle, needs 1 drop action at the goal
                base_cost = 1
            else:
                # Package is on the ground, needs 1 pick-up + 1 drop action
                base_cost = 2

            total_cost += base_cost + drive_cost

        return total_cost
