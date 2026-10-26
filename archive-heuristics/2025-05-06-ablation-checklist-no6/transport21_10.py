from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Ensure fact is treated as a string and remove outer parentheses
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
        # Handle unexpected formats, maybe return empty list or raise error
        return []

    # Split by whitespace
    return fact_str[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Must match the number of arguments exactly
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    Estimates the number of actions needed to move each package to its goal location.
    It sums the estimated costs for each package independently, ignoring vehicle capacity
    and coordination.

    # Heuristic Initialization
    - Builds a graph of the road network and computes all-pairs shortest paths using BFS.
    - Extracts the goal location for each package.
    - Identifies all packages and vehicles present in the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it on the ground at some location, or is it inside a vehicle?
    2. If the package is on the ground at location `L_current`:
       - It needs to be picked up (1 action).
       - It needs to be transported from `L_current` to its goal location `L_goal`. The estimated cost for this transport is the shortest path distance between `L_current` and `L_goal` in the road network (number of `drive` actions).
       - It needs to be dropped at `L_goal` (1 action).
       - Total estimated cost for this package: 1 + shortest_path_distance + 1.
    3. If the package is inside a vehicle, and the vehicle is at location `L_current`:
       - It needs to be transported from `L_current` to its goal location `L_goal`. The estimated cost for this transport is the shortest path distance between `L_current` and `L_goal` (number of `drive` actions).
       - It needs to be dropped at `L_goal` (1 action).
       - Total estimated cost for this package: shortest_path_distance + 1.
    4. Sum the estimated costs for all packages that are not at their goal location.
    5. If any package's goal location is unreachable from its current location (or its vehicle's location) in the road network, the heuristic returns infinity.
    6. The heuristic value is 0 if and only if all packages are at their goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the road graph, computing distances,
        and extracting package goal locations.
        """
        super().__init__(task)

        # Build the road graph from static facts
        self.road_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        self.locations = list(locations) # Store locations list

        # Compute all-pairs shortest paths using BFS
        self.dist = {} # Store shortest path distances: dist[start_loc][end_loc]
        for start_node in self.locations:
            self.dist[start_node] = {}
            # Initialize distances
            for loc in self.locations:
                self.dist[start_node][loc] = float('inf')
            self.dist[start_node][start_node] = 0

            # BFS
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                # Ensure the node exists in the graph before accessing neighbors
                if u in self.road_graph:
                    for v in self.road_graph[u]:
                        if self.dist[start_node][v] == float('inf'):
                            self.dist[start_node][v] = self.dist[start_node][u] + 1
                            queue.append(v)

        # Extract goal locations for packages
        self.goal_locations = {}
        # Assume goal facts are always (at package location) for packages.
        # Packages are identified by being the first argument of an 'at' predicate in the goal.
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location

        # Identify all packages and vehicles for efficient state parsing
        self.all_packages = set(self.goal_locations.keys())
        self.all_vehicles = set()
        # Scan initial state and static facts (like capacity) to find vehicles
        # Initial state is task.initial_state
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # If it's not a package we know about, assume it's a vehicle if it's 'at' a location
                 if obj not in self.all_packages:
                     self.all_vehicles.add(obj)
             elif match(fact, "in", "*", "*"):
                  # Package is in a vehicle in initial state
                  pkg, veh = get_parts(fact)[1:]
                  # Add package if not already known (e.g., if goal is implicit)
                  # self.all_packages.add(pkg) # Assuming packages are defined by goals
                  self.all_vehicles.add(veh)
             elif match(fact, "capacity", "*", "*"):
                  veh, size = get_parts(fact)[1:]
                  self.all_vehicles.add(veh)

        # Also check static facts for capacity definitions
        for fact in self.static:
             if match(fact, "capacity", "*", "*"):
                  veh, size = get_parts(fact)[1:]
                  self.all_vehicles.add(veh)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state
        total_cost = 0

        # Parse current state to find locations of packages and vehicles
        pkg_at = {}
        pkg_in = {}
        veh_at = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "at" and len(args) == 2:
                obj, loc = args
                if obj in self.all_packages:
                    pkg_at[obj] = loc
                elif obj in self.all_vehicles:
                    veh_at[obj] = loc
            elif predicate == "in" and len(args) == 2:
                pkg, veh = args
                # Ensure both package and vehicle are known types
                if pkg in self.all_packages and veh in self.all_vehicles:
                     pkg_in[pkg] = veh
            # Ignore other predicates like capacity, capacity-predecessor, road etc.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal. Find its current location/status.
            current_location = None
            is_in_vehicle = False
            # vehicle_carrying = None # Not strictly needed for cost calculation here

            if package in pkg_at:
                current_location = pkg_at[package]
                is_in_vehicle = False
            elif package in pkg_in:
                vehicle_carrying = pkg_in[package]
                # Find the location of the vehicle
                if vehicle_carrying in veh_at:
                    current_location = veh_at[vehicle_carrying]
                    is_in_vehicle = True
                else:
                    # Vehicle location not found? This indicates an inconsistent state.
                    # Treat as unreachable.
                    return float('inf')
            else:
                 # Package is not at any location and not in any vehicle.
                 # This should not happen in a valid state where the package is not at its goal.
                 # Treat as unreachable.
                 return float('inf')


            # Calculate cost for this package
            # Ensure current_location and goal_location are valid and connected
            if current_location not in self.dist or goal_location not in self.dist.get(current_location, {}):
                 # Locations not in graph or goal unreachable from current location
                 return float('inf')

            drive_cost = self.dist[current_location][goal_location]

            if drive_cost == float('inf'):
                # Goal is unreachable from current location in the road network
                return float('inf')

            if not is_in_vehicle:
                # Package is on the ground at current_location, needs pick-up, drive, drop
                total_cost += 1 # pick-up
                total_cost += drive_cost # drive
                total_cost += 1 # drop
            else: # Package is in a vehicle at current_location, needs drive, drop
                total_cost += drive_cost # drive
                total_cost += 1 # drop

        # If the loop completes, all packages with goals are either at their goal
        # or their costs have been summed. If total_cost is 0, all were at goal.
        # If total_cost is > 0, there are packages not at goal.
        # If any package was unreachable, we would have returned inf.
        # This ensures h=0 iff goal state and h=inf iff unreachable.
        # For solvable states, h is finite.

        return total_cost
