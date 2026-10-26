import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    from its current location to its goal location. It sums the estimated costs
    for each package independently, ignoring vehicle capacity and coordination
    constraints. The cost for a package includes pick-up (if on the ground),
    driving distance (shortest path on the road network), and drop-off.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles are available to pick up and drop off packages.
    - Vehicles can travel between any two connected locations on the road network.
    - Vehicle capacity is not explicitly modeled in the cost calculation.
    - The cost of each action (pick-up, drop, drive) is 1.
    - Object types (vehicles, packages) are inferred based on predicate usage
      in static facts and goals.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Infers object types (vehicles, packages) based on predicate usage in static facts and goals.
    - Builds a graph representation of the road network from static facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package that has a goal location.
       A package can be either on the ground at a location `(at package location)`
       or inside a vehicle `(in package vehicle)`. If inside a vehicle, find the
       vehicle's current location `(at vehicle vehicle_location)`.
    2. For each package `p` that is not yet satisfying its goal predicate `(at p goal_loc(p))`:
       a. If `p` is on the ground at `current_loc`:
          - The estimated cost for this package is 1 (pick-up) + shortest_path_distance(`current_loc`, `goal_loc(p)`) (drive) + 1 (drop).
       b. If `p` is inside a vehicle `v` which is at `vehicle_loc`:
          - The estimated cost for this package is shortest_path_distance(`vehicle_loc`, `goal_loc(p)`) (drive) + 1 (drop).
       c. If the goal location is unreachable from the package's current position (or its vehicle's position), the heuristic value is infinity.
       d. If the package's state (location or vehicle) is unknown in the current state, the heuristic value is infinity.
    3. The total heuristic value for the state is the sum of the estimated costs for all packages that are not satisfying their goal predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, inferring object
        types, and building the road network graph for shortest path calculations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Infer object types (vehicles, packages) and collect locations
        self.vehicles = set()
        self.packages = set()
        locations_set = set() # Use a set initially to collect all locations

        # Process static facts to infer types and locations
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "capacity" and len(parts) == 3:
                self.vehicles.add(parts[1])
            elif predicate == "in" and len(parts) == 3:
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])
            elif predicate == "road" and len(parts) == 3:
                 locations_set.add(parts[1])
                 locations_set.add(parts[2])
            # 'at' facts in static are initial state, processed in __call__

        # Process goal facts to infer types and locations
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assume anything in an 'at' goal predicate is a package
                self.packages.add(obj)
                locations_set.add(loc)
            # No 'in' or 'capacity' in goal facts typically

        # 1. Extract goal locations for each package (only for identified packages)
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                 package, location = parts[1], parts[2]
                 if package in self.packages: # Ensure it's a package we care about
                    self.goal_locations[package] = location

        # 2. Build the road network graph
        self.road_graph = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.road_graph[loc1].add(loc2)

        self.locations = list(locations_set) # Convert set to list

        # 3. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            distances = self._bfs(start_node)
            for end_node, dist in distances.items():
                self.shortest_paths[(start_node, end_node)] = dist

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        if start_node not in distances:
             # Start node is not a known location from static facts/goals.
             # It cannot reach any known location.
             return {} # Return empty distances

        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Only explore if current_node is in the graph (has outgoing edges)
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    # Ensure neighbor is a known location before updating distance
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings.

        # 1. Identify current locations of packages and vehicles from the current state
        package_locations = {} # {package: location} for packages on ground
        package_in_vehicle = {} # {package: vehicle} for packages in vehicles
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.vehicles:
                     vehicle_locations[obj] = loc
                elif obj in self.packages:
                     package_locations[obj] = loc # Package is on the ground
                # Ignore other 'at' facts if any
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Only track 'in' for known packages and vehicles
                if package in self.packages and vehicle in self.vehicles:
                    package_in_vehicle[package] = vehicle
            # Ignore other predicates like 'capacity', 'capacity-predecessor'

        total_cost = 0

        # 2. Calculate cost for each package not satisfying its goal predicate
        for package, goal_loc in self.goal_locations.items():
            # Check if package is already satisfying the goal predicate (at package goal_loc)
            goal_fact_str = f"(at {package} {goal_loc})"
            if goal_fact_str in state:
                continue # Package is already at the goal location predicate

            # Package is not satisfying the goal predicate. Find its current state.
            is_on_ground = package in package_locations
            is_in_vehicle = package in package_in_vehicle

            if is_on_ground and is_in_vehicle:
                 # This state is inconsistent (package cannot be both at a location and in a vehicle)
                 return float('inf') # Indicate an invalid state or unreachable goal from here.
            elif not is_on_ground and not is_in_vehicle:
                 # Package location/state is unknown in the current state facts
                 return float('inf') # Indicate unreachable.
            elif is_on_ground:
                # Package is on the ground at current_loc
                current_loc = package_locations[package]
                # Ensure current_loc is a known location in our graph
                if current_loc not in self.locations:
                     return float('inf') # Cannot compute path from unknown location

                dist = self.shortest_paths.get((current_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    # Goal is unreachable from package's current location
                    return float('inf')
                # Cost = 1 (pick-up) + dist (drive) + 1 (drop)
                total_cost += 2 + dist

            elif is_in_vehicle:
                # Package is in a vehicle at vehicle_loc
                vehicle = package_in_vehicle[package]
                vehicle_loc = vehicle_locations.get(vehicle)

                if vehicle_loc is None:
                     # Vehicle location is unknown in the current state facts
                     return float('inf') # Indicate unreachable.

                # Ensure vehicle_loc is a known location in our graph
                if vehicle_loc not in self.locations:
                     return float('inf') # Cannot compute path from unknown location

                dist = self.shortest_paths.get((vehicle_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    # Goal is unreachable from vehicle's current location
                    return float('inf')
                # Cost = dist (drive) + 1 (drop)
                total_cost += 1 + dist

        return total_cost
