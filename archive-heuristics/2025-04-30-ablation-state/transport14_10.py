from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque # Using deque for efficient queue operations in BFS

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total minimum actions needed to move all packages
    to their goal locations. It sums the estimated costs for each package
    independently, considering its current state (at a location or in a vehicle)
    and the shortest road distance to its goal location.

    # Assumptions
    - The road network defined by `(road l1 l2)` facts is the only way vehicles can move between locations.
    - All locations relevant to package goals and initial positions are part of the road network graph.
    - Shortest paths in the road network represent the minimum number of `drive` actions.
    - Any vehicle can be used to transport any package (capacity and vehicle availability are relaxed).
    - A package on the ground needs 1 pick-up and 1 drop action.
    - A package in a vehicle needs 1 drop action.
    - The cost of moving a package is the sum of load/unload actions and the shortest path distance the vehicle must travel *while carrying the package* (or to get to the package's location).

    # Heuristic Initialization
    - Parses goal facts (`(at ?p ?l)`) to map each package to its target location.
    - Parses static facts (`road`, `capacity`) to build the road network graph and identify vehicles.
    - Computes all-pairs shortest paths between locations using BFS from each location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the current location of all packages that have a goal, and the current location of all vehicles.
       - A package is either `(at ?p ?l)` or `(in ?p ?v)`.
       - A vehicle is `(at ?v ?l)`.
    3. For each package that has a goal location:
       a. Check if the package is currently at its goal location. If yes, the cost for this package is 0; proceed to the next package.
       b. If the package is not at its goal:
          i. If the package is currently on the ground at `current_location` (`(at package current_location)`):
             - Find the shortest distance `d` from `current_location` to the package's `goal_location` using the precomputed shortest paths.
             - If no path exists (`d` is None), the state is likely unsolvable; return infinity.
             - Otherwise, add `1` (for pick-up) + `d` (for driving) + `1` (for drop) to the total cost.
          ii. If the package is currently inside a vehicle `vehicle` (`(in package vehicle)`):
             - Find the current location `vehicle_location` of that `vehicle` (`(at vehicle vehicle_location)`).
             - If the vehicle's location is not found, the state is likely invalid/unsolvable; return infinity.
             - Find the shortest distance `d` from `vehicle_location` to the package's `goal_location`.
             - If no path exists (`d` is None), the state is likely unsolvable; return infinity.
             - Otherwise, add `d` (for driving) + `1` (for drop) to the total cost.
    4. Return the accumulated total cost. If any step returned infinity, that value propagates.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) >= 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # 2. Build the road graph, find all locations, and identify vehicles
        self.road_graph = {} # { loc: [neighbor1, neighbor2, ...] }
        locations = set()
        self.vehicles = set() # Store vehicle names

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "road" and len(parts) >= 3:
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1)
            elif parts[0] == "capacity" and len(parts) >= 3:
                 vehicle = parts[1]
                 self.vehicles.add(vehicle)

        self.locations = list(locations) # Store locations as a list

        # 3. Compute all-pairs shortest paths using BFS from each location
        self.shortest_paths = {} # { (loc_from, loc_to): distance }

        for start_node in self.locations:
            # Perform BFS starting from start_node
            queue = deque([(start_node, 0)]) # (current_location, distance)
            visited = {start_node}
            self.shortest_paths[(start_node, start_node)] = 0 # Distance to self is 0

            while queue:
                (current_loc, dist) = queue.popleft() # Use popleft for efficient queue

                # Explore neighbors
                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.shortest_paths[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def get_shortest_distance(self, loc_from, loc_to):
        """Helper to retrieve precomputed shortest distance."""
        # Check if both locations are known in the graph
        if loc_from not in self.locations or loc_to not in self.locations:
             # This indicates an issue with the problem definition or state.
             # For heuristic purposes, treat as unreachable.
             return None

        # Distance is already computed and stored
        # Returns None if no path found (shouldn't happen if graph is connected and includes all relevant locations)
        return self.shortest_paths.get((loc_from, loc_to))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_current_state = {} # { package_name: ('at', loc) or ('in', vehicle) }
        vehicle_current_location = {} # { vehicle_name: loc }

        # Parse the state specifically for package and vehicle locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we care about
                     package_current_state[obj] = ('at', loc)
                elif obj in self.vehicles: # It's a vehicle we identified statically
                     vehicle_current_location[obj] = loc

            elif predicate == "in" and len(parts) >= 3:
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # It's a package we care about
                    package_current_state[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            current_state_info = package_current_state.get(package)

            # If package is not mentioned in state facts (shouldn't happen in valid PDDL)
            if current_state_info is None:
                 # Treat as if it's not at the goal and cannot be moved.
                 return float('inf')

            state_type, current_loc_or_vehicle = current_state_info

            # Check if package is already at the goal location
            if state_type == 'at' and current_loc_or_vehicle == goal_location:
                continue # Package is already at goal, cost is 0 for this package.

            # If not at goal, calculate cost
            if state_type == 'in':
                # Package is in a vehicle. Find vehicle location.
                vehicle = current_loc_or_vehicle
                loc_v_current = vehicle_current_location.get(vehicle)

                if loc_v_current is None:
                     # Vehicle location not found, invalid state?
                     return float('inf')

                # Cost: drive vehicle from its current location to package's goal location + drop package
                distance = self.get_shortest_distance(loc_v_current, goal_location)

                if distance is None:
                    # Goal location is unreachable from vehicle's current location via road network.
                    return float('inf')

                total_cost += distance + 1 # 1 for drop action

            elif state_type == 'at': # Package is 'at' a location, not the goal
                current_location = current_loc_or_vehicle

                # Cost: pickup package + drive vehicle from package location to package's goal location + drop package
                distance = self.get_shortest_distance(current_location, goal_location)

                if distance is None:
                    # Goal location is unreachable from package's current location via road network.
                    return float('inf')

                total_cost += 1 + distance + 1 # 1 for pickup, 1 for drop

        return total_cost
