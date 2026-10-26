# Assume the Heuristic base class is provided in the environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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

def bfs(graph, start):
    """Computes shortest path distances from start to all reachable nodes."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        distance = distances[current]
        # Ensure current node exists in graph keys before iterating neighbors
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph, all_nodes):
    """Computes shortest path distances between all pairs of nodes."""
    all_distances = {}
    # Run BFS starting from every node in the collected set of all_nodes.
    for start_node in all_nodes:
        all_distances[start_node] = bfs(graph, start_node)
    return all_distances


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, and drive steps)
    required to move each package to its goal location, considering each package
    independently and ignoring vehicle capacity and availability constraints.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicle capacity and availability are ignored (relaxed problem).
    - Any location is reachable from any other location within the same connected component
      of the road network. The cost of driving is the shortest path distance in the road network.
    - Picking up a package costs 1 action.
    - Dropping a package costs 1 action.
    - Driving between adjacent locations costs 1 action.
    - If a goal location is unreachable from a package's current location (or its vehicle's location)
      via the road network, a large penalty is added for that package.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build a graph representing the road network from the static facts.
    - Collect all locations mentioned in road facts and goal facts.
    - Compute the shortest path distance between all pairs of these collected locations using BFS on the road network graph.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that is not yet at its goal location `goal_l`:
    1. Determine the package's current status by examining the state:
       - Is it on the ground at some location `current_l` (fact `(at p current_l)`)?
       - Is it inside a vehicle `v` (fact `(in p v)`)? If so, find the vehicle's location `vehicle_l` (fact `(at v vehicle_l)`).
    2. Calculate the minimum actions required based on the current status, ignoring vehicle constraints:
       - If `p` is already at its goal location `goal_l` (fact `(at p goal_l)`):
         - Cost for this package is 0.
       - If `p` is on the ground at `current_l` (`current_l != goal_l`):
         - It needs to be picked up (1 action).
         - It needs to be transported from `current_l` to `goal_l`. The minimum drive actions is the shortest path distance `dist(current_l, goal_l)`.
         - It needs to be dropped at `goal_l` (1 action).
         - Total cost for this package: 1 (pick) + `dist(current_l, goal_l)` (drive) + 1 (drop).
       - If `p` is inside vehicle `v` which is at `vehicle_l`:
         - If `vehicle_l == goal_l`:
           - It needs to be dropped at `goal_l` (1 action).
           - Total cost for this package: 1 (drop).
         - If `vehicle_l != goal_l`:
           - It needs to be transported from `vehicle_l` to `goal_l`. Minimum drive actions: `dist(vehicle_l, goal_l)`.
           - It needs to be dropped at `goal_l` (1 action).
           - Total cost for this package: `dist(vehicle_l, goal_l)` (drive) + 1 (drop).
       - If the package's location or vehicle's location is unknown or leads to an unreachable goal location via the road network, a large penalty is added for this package's cost.
    3. The total heuristic value is the sum of the minimum costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and pre-computing shortest path distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # else: ignore malformed goal fact or non-'at' goal fact

        # Build the road network graph and collect all relevant locations.
        self.road_graph = {}
        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                _, loc1, loc2 = parts
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = set()
                self.road_graph[loc1].add(loc2)
                # Assuming roads are bidirectional unless specified otherwise
                if loc2 not in self.road_graph:
                    self.road_graph[loc2] = set()
                self.road_graph[loc2].add(loc1)

        # Add locations from goals that might not be in road facts
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "at" and len(parts) == 3:
                 _, _, loc = parts
                 all_locations.add(loc)

        # Ensure all collected locations are keys in the graph, even if they have no roads
        # This is important so BFS can be called starting from any relevant location.
        for loc in all_locations:
             if loc not in self.road_graph:
                  self.road_graph[loc] = set()

        # Compute all-pairs shortest paths using all collected locations.
        self.shortest_paths = compute_all_pairs_shortest_paths(self.road_graph, all_locations)


    def get_distance(self, loc1, loc2):
        """
        Looks up the shortest path distance between loc1 and loc2.
        Returns the distance or None if unreachable or location not in the pre-computed paths.
        """
        if loc1 == loc2:
            return 0
        # Check if loc1 was a starting point for BFS and if loc2 was reached from loc1
        if loc1 in self.shortest_paths and loc2 in self.shortest_paths[loc1]:
            return self.shortest_paths[loc1][loc2]
        return None # Indicates unreachable or location not in the graph/paths


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached. If so, heuristic is 0.
        # This check is important for correctness, especially for GBFS.
        # Although the planner might do this before calling the heuristic on the goal node,
        # including it here makes the heuristic self-contained and correct for goal states.
        if self.goals <= state:
             return 0

        # Collect current locations and vehicle contents efficiently
        current_locatables_at = {} # Maps locatable (pkg or veh) to location
        package_in_vehicle_map = {} # Maps package to vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locatables_at[obj] = loc
            elif parts and parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle_map[package] = vehicle

        total_cost = 0
        UNREACHABLE_PENALTY = 1000 # Large cost for unreachable goals per package

        for package, goal_location in self.goal_locations.items():
            package_cost = 0

            # Check if package is already at goal
            if package in current_locatables_at and current_locatables_at[package] == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal. Where is it?
            if package in package_in_vehicle_map:
                # Package is in a vehicle
                vehicle = package_in_vehicle_map[package]
                if vehicle in current_locatables_at:
                    vehicle_location = current_locatables_at[vehicle]
                    # Cost = drive from vehicle_location to goal_location + drop
                    dist = self.get_distance(vehicle_location, goal_location)
                    if dist is None: # Unreachable via road network from vehicle's current location
                         package_cost += UNREACHABLE_PENALTY
                    else:
                        package_cost += dist # Drive cost
                        package_cost += 1 # Drop cost
                else:
                    # Vehicle location unknown - should not happen in valid state
                    package_cost += UNREACHABLE_PENALTY
            elif package in current_locatables_at:
                # Package is on the ground, not at goal
                current_location = current_locatables_at[package]
                # Cost = pick-up + drive from current_location to goal_location + drop
                dist = self.get_distance(current_location, goal_location)
                if dist is None: # Unreachable via road network from package's current location
                    package_cost += UNREACHABLE_PENALTY
                else:
                    package_cost += 1 # Pick-up cost
                    package_cost += dist # Drive cost
                    package_cost += 1 # Drop cost
            else:
                 # Package location unknown - should not happen in valid state
                 package_cost += UNREACHABLE_PENALTY

            total_cost += package_cost

        return total_cost
