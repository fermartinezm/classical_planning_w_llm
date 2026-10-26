from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, although assuming valid PDDL fact strings
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport each
    misplaced package to its goal location. It sums the estimated costs for
    each package independently, ignoring vehicle capacity constraints and
    vehicle availability at pickup locations.

    # Assumptions
    - Drive actions between connected locations cost 1.
    - The road network is undirected (if road A-B exists, B-A exists).
    - Vehicle capacity is ignored.
    - Any vehicle can be used for any package transport (implicitly, as vehicle identity is not tracked per package transport estimate).
    - The cost for a vehicle to reach a package's initial ground location
      is ignored; it's assumed a vehicle is available for pickup instantly.
    - Packages already at their goal location but inside a vehicle need 1 drop action.
    - The road network is connected, or packages only need to move between connected components. Unreachable goals result in infinite heuristic.

    # Heuristic Initialization
    - Identify all locations from static road facts and 'at' predicates.
    - Build an undirected graph representing the road network using 'road' facts.
    - Compute all-pairs shortest paths (distances) between locations using BFS.
    - Identify all packages and their goal locations from the task goals.
    - Identify all packages, vehicles, and locations present in the initial state, goals, and static facts to correctly interpret the state during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Determine the current status of each package that has a goal location: Is it on the ground at a location, or is it inside a vehicle? If inside a vehicle, find the vehicle's current location. This gives the effective current location for each package.
    2. Initialize total heuristic cost to 0.
    3. For each package that has a goal location:
       - Get its goal location.
       - Get its effective current location and whether it's currently in a vehicle.
       - If the effective current location is the same as the goal location:
         - If the package is *inside* a vehicle, it still needs to be dropped. Add 1 to the total cost.
         - If the package is on the ground, it's already at its final goal state for this package. Add 0.
       - If the effective current location is different from the goal location:
         - Calculate the shortest distance between the effective current location and the goal location using the precomputed distance map. If unreachable, return infinity.
         - If the package is currently on the ground: Add 1 (pick-up) + distance (drive) + 1 (drop) to the total cost.
         - If the package is currently inside a vehicle: Add distance (drive) + 1 (drop) to the total cost.
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify object types (packages, vehicles, locations, sizes)
        # Infer types from predicate usage across initial state, goals, and static facts
        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.sizes = set()

        all_facts = list(initial_state) + list(self.goals) + list(static_facts)

        locatables_from_at = set()
        packages_from_in = set()
        vehicles_from_in = set()
        vehicles_from_capacity = set()
        sizes_from_capacity = set()
        sizes_from_capacity_predecessor = set()
        locations_from_road = set()
        locations_from_at = set()


        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'road' and len(args) == 2:
                locations_from_road.update(args)
            elif predicate == 'at' and len(args) == 2:
                locatables_from_at.add(args[0])
                locations_from_at.add(args[1])
            elif predicate == 'in' and len(args) == 2:
                packages_from_in.add(args[0])
                vehicles_from_in.add(args[1])
            elif predicate == 'capacity' and len(args) == 2:
                 vehicles_from_capacity.add(args[0])
                 sizes_from_capacity.add(args[1])
            elif predicate == 'capacity-predecessor' and len(args) == 2:
                 sizes_from_capacity_predecessor.update(args)

        self.locations = locations_from_road | locations_from_at
        self.sizes = sizes_from_capacity | sizes_from_capacity_predecessor
        self.packages = packages_from_in

        # Vehicles are locatables that are not packages, plus those in 'in' or 'capacity'
        potential_vehicles = vehicles_from_in | vehicles_from_capacity
        # Any locatable (from 'at') that wasn't identified as a package must be a vehicle
        vehicles_from_locatables = locatables_from_at - self.packages
        self.vehicles = potential_vehicles | vehicles_from_locatables


        # 2. Build road graph
        self.road_graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                if l1 in self.road_graph and l2 in self.road_graph: # Ensure locations are known
                    self.road_graph[l1].append(l2)
                    self.road_graph[l2].append(l1) # Roads are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.distance_map = {}
        for start_node in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                u = queue.popleft()
                current_dist = distances[u]

                # Check if u is a valid location with roads defined
                if u in self.road_graph:
                    for v in self.road_graph.get(u, []): # Use .get for safety
                        if v in distances and distances[v] == float('inf'): # Ensure v is a known location
                            distances[v] = current_dist + 1
                            queue.append(v)
            self.distance_map[start_node] = distances

        # 4. Store goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Only consider goals for objects identified as packages
                if package in self.packages:
                    self.package_goals[package] = location


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Determine current status and effective location for each package
        package_current_info = {} # package -> {'location': loc, 'in_vehicle': bool, 'vehicle': v or None}
        vehicle_locations = {}    # vehicle -> location

        # First pass: find direct locations (at) and packages in vehicles (in)
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at' and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj in self.packages:
                    package_current_info[obj] = {'location': loc, 'in_vehicle': False, 'vehicle': None}
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif predicate == 'in' and len(args) == 2:
                p, v = args[0], args[1]
                if p in self.packages and v in self.vehicles:
                     package_current_info[p] = {'vehicle': v, 'in_vehicle': True, 'location': None} # Location will be updated

        # Second pass: update location for packages that are in vehicles
        # Need to iterate over packages that we know are in vehicles
        packages_in_vehicles_to_update = [
            p for p, info in package_current_info.items() if info['in_vehicle']
        ]

        for p in packages_in_vehicles_to_update:
             info = package_current_info[p]
             v = info['vehicle']
             if v in vehicle_locations:
                 info['location'] = vehicle_locations[v]
             else:
                 # Vehicle location not found for a package in a vehicle.
                 # This state is likely invalid or incomplete based on domain rules.
                 # Treat as unreachable for this package.
                 info['location'] = 'unreachable' # Use a special marker


        # Calculate total cost
        total_cost = 0

        for package, goal_loc in self.package_goals.items():
            if package not in package_current_info:
                 # Package with a goal is not mentioned in 'at' or 'in' facts in the state.
                 # This indicates an invalid state or missing info.
                 # Treat as unreachable.
                 return float('inf')

            current_info = package_current_info[package]
            current_loc = current_info.get('location')

            if current_loc is None or current_loc == 'unreachable':
                 # Location could not be determined (e.g., vehicle location missing)
                 return float('inf')

            # Check if package is already at goal location
            if current_loc == goal_loc:
                # If at goal, check if it needs dropping
                if current_info['in_vehicle']:
                    total_cost += 1 # Needs a drop action
                # If on ground at goal, cost is 0 for this package
            else:
                # Package is not at goal location
                # Check if distance is computable
                if current_loc not in self.distance_map or goal_loc not in self.distance_map[current_loc]:
                     # Goal location unreachable from current location
                     return float('inf')

                dist = self.distance_map[current_loc][goal_loc]

                if dist == float('inf'):
                     # Goal location is unreachable
                     return float('inf')


                if current_info['in_vehicle']:
                    # Package is in a vehicle, needs drive and drop
                    total_cost += dist + 1
                else:
                    # Package is on the ground, needs pick-up, drive, and drop
                    total_cost += 1 + dist + 1

        # Heuristic must be 0 only at goal state.
        # If total_cost is 0, it means for every package with a goal:
        # - current_loc == goal_loc AND not current_info['in_vehicle']
        # This matches the goal state for packages (assuming goals are only (at ?p ?l)).
        # If a package is at goal location but in vehicle, cost is 1, so total_cost > 0.
        # This seems correct.

        return total_cost
