from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # This simple check assumes the number of parts in the fact matches the number of args in the pattern
    # or that fnmatch handles wildcards appropriately across length differences (it doesn't for zip).
    # A more robust check would align parts and args considering wildcards.
    # However, for typical PDDL predicates and patterns like ("at", "*", "loc"), this is fine.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# BFS helper
def bfs_distances(graph, start_node):
    """Compute shortest distances from start_node to all reachable nodes in a graph."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# Heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity
    and the possibility of carrying multiple packages simultaneously. It sums
    the estimated costs for each package that is not yet at its goal.

    # Assumptions
    - The road network is static and provides directed connections between locations.
    - Packages are either on the ground at a location or inside a vehicle.
    - Vehicles are always located at a specific location.
    - Vehicle capacity constraints are ignored.
    - The possibility of a vehicle carrying multiple packages on a single trip is ignored;
      each package's transport is estimated independently.
    - All locations relevant to package goals and initial positions are part of the
      static road network or reachable from it.
    - The cost of any action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Parses the goal conditions to identify the target location for each package.
    - Parses the static facts to build the road network graph.
    - Computes all-pairs shortest path distances between all relevant locations
      using Breadth-First Search (BFS) on the road graph.
    - Identifies packages and vehicles present in the initial state and goals
      to distinguish object types.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle. A package can be
       either on the ground at a location `(at p l)` or inside a vehicle `(in p v)`,
       in which case its effective location is the location of the vehicle `(at v l_v)`.
    2. Initialize the total estimated cost to 0.
    3. For each package `p` that has a goal location `loc_goal`:
       a. Check if the package is already at its goal location `(at p loc_goal)`. If yes,
          add 0 to the total cost for this package and proceed to the next package.
       b. If the package is not at its goal:
          i. Determine the package's current effective location.
             - If `(at p loc_current)` is true, the current location is `loc_current`.
             - If `(in p v)` is true and `(at v loc_v)` is true, the current location is `loc_v`.
          ii. Estimate the cost to move this package to its goal location:
              - If the package is on the ground at `loc_current`:
                Estimated cost = 1 (pick-up) + shortest_distance(`loc_current`, `loc_goal`) + 1 (drop).
              - If the package is inside a vehicle at `loc_v`:
                Estimated cost = shortest_distance(`loc_v`, `loc_goal`) + 1 (drop).
              - If the goal location is unreachable from the current effective location
                via the road network, the state is likely unsolvable or requires
                actions not considered; return infinity.
          iii. Add the estimated cost for this package to the total cost.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and computing shortest path distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to identify objects

        self.goal_locations = {}
        self.road_graph = {}
        locations = set()
        self.packages = set()
        self.vehicles = set()

        # 1. Identify Packages and Vehicles from initial state and goals
        all_objects = set()
        for fact in initial_state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "in":
                self.packages.add(parts[1])
                self.vehicles.add(parts[2])
                all_objects.add(parts[1])
                all_objects.add(parts[2])
            elif predicate == "capacity":
                self.vehicles.add(parts[1])
                all_objects.add(parts[1])
            elif predicate == "at":
                 all_objects.add(parts[1])
                 locations.add(parts[2]) # Collect initial locations

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
                self.packages.add(package)
                all_objects.add(package)
                locations.add(location) # Collect goal locations

        # Any object seen in initial state/goals that isn't a package must be a vehicle
        # This handles vehicles only mentioned in 'at' facts initially.
        self.vehicles.update(all_objects - self.packages)


        # 2. Build Road Graph from static facts
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.road_graph.setdefault(l1, []).append(l2)
                locations.add(l1)
                locations.add(l2)
            # Ignore capacity-predecessor for this heuristic

        # Ensure all relevant locations (initial, goal, road endpoints) are considered for BFS
        all_relevant_locations = locations

        # 3. Compute All-Pairs Shortest Paths
        self.distances = {}
        for start_loc in all_relevant_locations:
             self.distances[start_loc] = bfs_distances(self.road_graph, start_loc)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Track current locations/containment for packages and vehicles
        package_current_info = {} # Maps package -> ('at', loc) or ('in', vehicle)
        vehicle_locations = {} # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_info[obj] = ('at', loc)
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                # Ensure package and vehicle are known types (should be from init)
                if package in self.packages and vehicle in self.vehicles:
                     package_current_info[package] = ('in', vehicle)
                # else: print(f"Warning: Unknown object types in 'in' fact: {fact}")


        # Compute cost for each package that needs to reach its goal
        for package, loc_goal in self.goal_locations.items():
            # Check if the package is already at the goal location on the ground
            if package_current_info.get(package) == ('at', loc_goal):
                continue # Goal met for this package

            # Package is not at the goal location on the ground.
            # Find its current effective location.
            current_info = package_current_info.get(package)

            if current_info is None:
                 # This package is in the goal but not found in the current state's
                 # 'at' or 'in' facts. This indicates an invalid state representation
                 # or an unreachable goal due to the package disappearing.
                 # Return infinity as it's likely unsolvable from this state.
                 # print(f"Error: Goal package {package} not found in state.")
                 return float('inf')

            current_type, current_loc_or_vehicle = current_info

            if current_type == 'at':
                # Package is on the ground at current_loc_or_vehicle
                loc_current = current_loc_or_vehicle
                # Cost: pick + drive + drop
                # Need distance from loc_current to loc_goal
                dist = self.distances.get(loc_current, {}).get(loc_goal)
                if dist is None:
                     # Goal location unreachable from current location
                     # print(f"Error: Goal location {loc_goal} unreachable from {loc_current} for package {package}.")
                     return float('inf')
                total_cost += 1 + dist + 1 # pick (1) + drive (dist) + drop (1)

            elif current_type == 'in':
                # Package is in vehicle current_loc_or_vehicle
                vehicle = current_loc_or_vehicle
                loc_v = vehicle_locations.get(vehicle)

                if loc_v is None:
                    # Vehicle location not found? Should not happen in valid state.
                    # print(f"Error: Vehicle {vehicle} carrying {package} not found at a location.")
                    return float('inf')

                # Cost: drive + drop
                # Need distance from vehicle's location loc_v to loc_goal
                dist = self.distances.get(loc_v, {}).get(loc_goal)
                if dist is None:
                     # Goal location unreachable from vehicle's current location
                     # print(f"Error: Goal location {loc_goal} unreachable from vehicle location {loc_v} for package {package}.")
                     return float('inf')

                # If the vehicle is already at the goal location, only drop is needed.
                # distance(loc_v, loc_goal) will be 0 in this case.
                total_cost += dist + 1 # drive (dist) + drop (1)

        return total_cost
