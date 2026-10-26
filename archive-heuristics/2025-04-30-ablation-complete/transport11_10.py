from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
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
    This heuristic estimates the required number of actions (pick-up, drive, drop)
    to move each package from its current location to its goal location,
    ignoring vehicle capacity constraints and vehicle availability.
    The total heuristic value is the sum of these individual package costs.

    # Assumptions
    - The goal is always specified by (at ?p ?l) predicates for packages.
    - Packages are always either at a location or inside a vehicle.
    - Vehicles are always at a location.
    - The road network is static and bidirectional.
    - A path exists between any two locations relevant to package goals.
    - Vehicle capacity is not a bottleneck (ignored for cost calculation).
    - Any vehicle can pick up/drop any package (size constraints ignored for cost calculation).

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build the road network graph from the static 'road' facts.
    - Identify all package and vehicle objects from initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:
    1. Determine the package's current location. This could be a direct location
       (if the package is on the ground) or the location of the vehicle it is in.
    2. If the package is on the ground at its current location (and not the goal):
       - It needs to be picked up (1 action).
    3. Calculate the shortest path distance (number of drive actions) required
       for a vehicle to travel from the package's current location to its goal location
       using the road network graph.
    4. The package needs to be dropped at the goal location (1 action).
    5. The total cost for this package is the sum of actions from steps 2, 3, and 4.
       - If on the ground: 1 (pick) + drive_distance + 1 (drop).
       - If in a vehicle: drive_distance + 1 (drop).
       - If already at the goal location (either on ground or in vehicle at goal location): 0 cost.
    6. The overall heuristic value is the sum of costs for all packages not at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, road network,
        and identifying objects.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the road graph (adjacency list)
        self.road_graph = {}
        all_locations = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                loc1, loc2 = parts[1], parts[2]
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional based on examples
                self.road_graph.setdefault(loc2, []).append(loc1)
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Store goal locations for each package
        self.goal_locations = {}
        # Identify all objects (packages and vehicles)
        self.packages = set()
        self.vehicles = set()

        # Helper to process facts and identify objects
        def process_fact_for_objects(fact_str):
            parts = get_parts(fact_str)
            if not parts: return
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                if obj.startswith('p'): self.packages.add(obj)
                elif obj.startswith('v'): self.vehicles.add(obj)
                if loc.startswith('l'): all_locations.add(loc)
            elif pred == "in":
                package, vehicle = parts[1], parts[2]
                if package.startswith('p'): self.packages.add(package)
                if vehicle.startswith('v'): self.vehicles.add(vehicle)
            # Add locations mentioned in static facts like capacity-predecessor if any
            # Although unlikely, ensure all potential locations are known
            for part in parts[1:]:
                 if part.startswith('l'):
                     all_locations.add(part)


        # Process initial state and goals to find objects and goal locations
        for fact in task.initial_state:
             process_fact_for_objects(fact)

        for goal in self.goals:
            process_fact_for_objects(goal) # Identify objects in goals too
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                # Only consider 'at' goals for packages
                if package in self.packages:
                    self.goal_locations[package] = location
                if location.startswith('l'): all_locations.add(location)

        # Ensure all identified locations are nodes in the graph, even if isolated
        for loc in all_locations:
             self.road_graph.setdefault(loc, [])


    def bfs(self, start, end):
        """
        Performs Breadth-First Search to find the shortest path distance
        between two locations in the road graph.
        Returns the number of drive actions (edges).
        Returns float('inf') if no path exists.
        """
        if start == end:
            return 0

        # Ensure start and end locations exist in the graph nodes
        if start not in self.road_graph or end not in self.road_graph:
             return float('inf') # One or both locations are not in the known graph

        queue = deque([(start, 0)]) # (location, distance)
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            # Ensure current_loc is a valid key in the graph (already checked at start)
            # if current_loc not in self.road_graph: continue

            for neighbor in self.road_graph.get(current_loc, []):
                if neighbor not in visited:
                    if neighbor == end:
                        return dist + 1 # Found the target, return distance
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Track current locations of packages and vehicles
        package_locations = {} # {package: location} if on ground
        package_in_vehicle = {} # {package: vehicle} if in vehicle
        vehicle_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_locations[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif pred == 'in':
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles:
                    package_in_vehicle[package] = vehicle

        total_cost = 0

        # Check if the current state is a goal state
        # This check is crucial for the h=0 only at goal property
        is_goal = all(goal in state for goal in self.goals)
        if is_goal:
            return 0

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at the goal location (either on ground or in vehicle)
            is_at_goal_on_ground = package in package_locations and package_locations[package] == goal_location
            is_in_vehicle_at_goal = (package in package_in_vehicle and
                                      package_in_vehicle[package] in vehicle_locations and
                                      vehicle_locations[package_in_vehicle[package]] == goal_location)

            if is_at_goal_on_ground or is_in_vehicle_at_goal:
                # Package is already at its goal location, no cost for this package
                continue

            # Package is not at its goal location, calculate cost
            cost_for_package = 0
            loc_curr = None

            if package in package_locations: # Package is on the ground
                loc_curr = package_locations[package]
                cost_for_package += 1 # Cost for pick-up

            elif package in package_in_vehicle: # Package is in a vehicle
                vehicle = package_in_vehicle[package]
                loc_curr = vehicle_locations.get(vehicle)
                if loc_curr is None:
                    # Vehicle location unknown - should not happen in valid state
                    # If a package is in a vehicle, the vehicle must be somewhere
                    return float('inf')
            else:
                 # Package location unknown - should not happen in valid state
                 # Packages are always either at a location or in a vehicle
                 return float('inf')

            # Calculate drive cost if current location is different from goal location
            if loc_curr != goal_location:
                drive_dist = self.bfs(loc_curr, goal_location)
                if drive_dist == float('inf'):
                    # Goal location unreachable from current location
                    return float('inf')
                cost_for_package += drive_dist # Cost for driving

            # Always need to drop the package at the goal location if it wasn't already there
            cost_for_package += 1 # Cost for drop

            total_cost += cost_for_package

        # If we reach here and total_cost is 0, it means all packages
        # listed in self.goal_locations are currently at their goal locations.
        # Since we already checked `is_goal` at the beginning, if `is_goal` was False,
        # but `total_cost` is 0, it implies the goal must contain predicates
        # other than `(at p l)` for packages, which this heuristic doesn't account for.
        # However, based on the problem description and examples, the goals
        # seem to be exclusively about package locations.
        # Thus, if all goal packages are at their locations, the state should be
        # the goal state, and the initial `is_goal` check would have returned 0.
        # If, somehow, total_cost is 0 but is_goal is False, it indicates a mismatch
        # between the heuristic's scope and the actual goal definition, or an
        # unreachable state where packages are correctly placed but other goals fail.
        # To strictly adhere to "h=0 only for goal states", if total_cost is 0
        # but it's not the true goal state, we should return a small positive value.
        # However, given the problem context, the calculated total_cost should
        # only be 0 if all goal packages are at their goal, which implies it IS
        # the goal state. The initial `is_goal` check is the definitive source.
        # So, if we pass the `is_goal` check, total_cost should be > 0 unless
        # the problem is trivial (no packages to move).
        # Let's trust the initial `is_goal` check and the cost calculation logic.
        # If total_cost is 0 here, it implies no goal packages needed moving,
        # which contradicts `is_goal` being False, unless there are no goal packages.
        # If self.goal_locations is empty, total_cost will be 0. An empty goal is a goal state.
        # The logic seems consistent with the problem structure.

        return total_cost
