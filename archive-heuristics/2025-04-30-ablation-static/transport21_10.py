from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
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


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It considers the current state of the package (on ground
    or in a vehicle) and the shortest path distance on the road network.

    # Assumptions
    - The cost of driving between two connected locations is 1.
    - Pick-up and drop actions cost 1 each.
    - Vehicle capacity and availability are not explicitly modeled, assuming
      a suitable vehicle is available when needed.
    - The shortest path distance on the road network is a reasonable estimate
      for the number of drive actions required for a vehicle carrying the package.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Collects all relevant locations mentioned in the problem (initial state, goals, roads).
    - Builds a graph representation of the road network using collected locations and road facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that is not yet at its goal location `goal_loc`:
    1. Determine the current physical location of the package `p_loc`.
       - If `p` is on the ground at `current_loc` (`(at p current_loc)` is true in the state), then `p_loc = current_loc`.
       - If `p` is inside a vehicle `v` (`(in p v)` is true in the state), find the location of vehicle `v` (`(at v v_loc)` is true in the state), then `p_loc = v_loc`.
       - If the package's location cannot be determined (which should not happen in valid states), return infinity.
    2. If `p_loc` is the same as `goal_loc`:
       - If `p` is on the ground at `goal_loc` (`(at p goal_loc)` is true), the cost for this package is 0 (already handled by the initial check).
       - If `p` is inside a vehicle `v` which is at `goal_loc` (`(in p v)` and `(at v goal_loc)` are true), the package needs to be dropped. Cost for this package is 1 (drop action).
    3. If `p_loc` is different from `goal_loc`:
       - The package (or its vehicle) needs to travel from `p_loc` to `goal_loc`. The estimated number of drive actions is the shortest path distance `dist(p_loc, goal_loc)`.
       - If the goal location is unreachable from the current location via the road network, return infinity.
       - If `p` is on the ground at `p_loc` (`(at p p_loc)` is true), it needs to be picked up and dropped. Cost for this package is `dist(p_loc, goal_loc)` + 1 (pick-up) + 1 (drop).
       - If `p` is inside a vehicle `v` at `p_loc` (`(in p v)` and `(at v p_loc)` are true), it needs to be dropped after the vehicle arrives. Cost for this package is `dist(p_loc, goal_loc)` + 1 (drop).
    4. The total heuristic value is the sum of the costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and computing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Goal is (at package location)
                if len(args) == 2: # Ensure it has object and location
                    package, location = args
                    self.goal_locations[package] = location

        # Collect all locations mentioned in the problem (initial state, goals, roads).
        all_locations = set()
        for fact in initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "at":
                 # (at obj loc) - loc is a location
                 if len(args) == 2: # Ensure it has object and location
                     all_locations.add(args[1])
        for goal in self.goals:
             predicate, *args = get_parts(goal)
             if predicate == "at":
                 # (at obj loc) - loc is a location
                 if len(args) == 2: # Ensure it has object and location
                     all_locations.add(args[1])
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                # (road l1 l2) - l1 and l2 are locations
                if len(args) == 2: # Ensure it has two locations
                    all_locations.add(args[0])
                    all_locations.add(args[1])

        self.locations = list(all_locations) # Store locations

        # Build the road network graph using all collected locations.
        self.road_graph = {loc: set() for loc in self.locations} # Initialize with all locations
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                if len(args) == 2: # Ensure it has two locations
                    l1, l2 = args
                    # Ensure l1 and l2 are valid locations we collected
                    if l1 in self.road_graph and l2 in self.road_graph:
                        self.road_graph[l1].add(l2)
                        self.road_graph[l2].add(l1) # Roads are bidirectional

        # Compute all-pairs shortest paths using BFS.
        self.distance = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        in the road network graph using BFS.
        Returns a dictionary distance[l1][l2] = shortest_distance.
        Handles unreachable locations by not including them in the inner dict.
        """
        distance = {}
        for start_node in self.locations:
            distance[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            distance[start_node][start_node] = 0 # Distance to self is 0

            while q:
                current_loc, dist = q.popleft()

                # Only explore if the current_loc is in the graph (i.e., connected to something by a road)
                # Locations might exist but not be part of the road network graph if they only appear in 'at' facts.
                # BFS will naturally handle this by only exploring reachable nodes.
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distance[start_node][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

        return distance

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map locatable objects (packages, vehicles) to their current physical location.
        current_locations = {}
        package_in_vehicle = {} # Track which package is in which vehicle

        # First pass: Find locations of all objects (vehicles and packages on ground)
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                if len(args) == 2: # Ensure it has object and location
                    obj, location = args # obj is vehicle or package
                    current_locations[obj] = location

        # Second pass: Find which packages are in which vehicles
        for fact in state:
             predicate, *args = get_parts(fact)
             if predicate == "in":
                 if len(args) == 2: # Ensure it has package and vehicle
                    package, vehicle = args
                    package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground.
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at goal, cost is 0 for this package.

            # Find the current physical location of the package.
            p_loc = None
            is_in_vehicle = package in package_in_vehicle

            if is_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find the location of the vehicle the package is in.
                if vehicle in current_locations:
                     p_loc = current_locations[vehicle]
                # else: Vehicle location is not known via (at v loc). This state is problematic.
            else:
                # Package is on the ground. Find its location.
                if package in current_locations:
                    p_loc = current_locations[package]
                # else: Package location is not known via (at p loc). This state is problematic.

            # If p_loc is still None, it means the package is not at goal, not in vehicle,
            # and has no (at p loc) fact. This should not happen in a valid state graph
            # reachable from a valid initial state.
            if p_loc is None:
                 # Cannot estimate cost if package location is unknown.
                 # This indicates an invalid state representation or an unhandled case.
                 # Returning infinity makes this state unattractive to the search.
                 return float('inf')


            # If the package's effective location is the goal location, it only needs dropping if in vehicle.
            if p_loc == goal_location:
                 # We already handled the (at p goal_loc) case (cost 0).
                 # So, if p_loc == goal_loc and it's not (at p goal_loc), it must be (in p v) and (at v goal_loc).
                 # Cost is 1 (drop).
                 total_cost += 1
            else:
                # Package is not at the goal location.
                # It needs to be moved from p_loc to goal_loc.
                # Estimated drive cost is the shortest distance.
                # Check if distance is known (i.e., goal_location is reachable from p_loc).
                if p_loc not in self.distance or goal_location not in self.distance[p_loc]:
                    # Goal location is unreachable from current location. Problem likely unsolvable.
                    # Return infinity or a very large number.
                    return float('inf') # Unreachable goal

                drive_cost = self.distance[p_loc][goal_location]

                # Add costs for pick-up and drop actions.
                if is_in_vehicle:
                    # Package is already in a vehicle. Needs drive + drop.
                    total_cost += drive_cost + 1 # 1 for drop
                else:
                    # Package is on the ground. Needs pick-up + drive + drop.
                    total_cost += 1 + drive_cost + 1 # 1 for pick-up, 1 for drop

        # The heuristic is 0 if and only if all packages are at their goal location on the ground.
        # This is ensured by the initial check `if f"(at {package} {goal_location})" in state: continue`.
        # If all packages satisfy this, the loop finishes and total_cost remains 0.
        # If any package is (in p v) at goal_loc, its cost is 1, making total_cost > 0.

        return total_cost
