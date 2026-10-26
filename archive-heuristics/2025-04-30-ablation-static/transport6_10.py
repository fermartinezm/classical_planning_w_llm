from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import collections # For BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure correct splitting
    return fact.strip()[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to move
    each package to its goal location, summed over all packages. It considers
    whether a package is on the ground or in a vehicle and uses shortest path
    distances on the road network to estimate drive costs. Vehicle capacity
    and availability are ignored.

    # Assumptions
    - The goal for each package is a specific location on the ground (`(at package location)`).
    - Packages can be on the ground or inside a vehicle.
    - Vehicles move between locations connected by roads.
    - Vehicle capacity constraints are relaxed (any vehicle can pick up any package).
    - Vehicle availability is relaxed (a vehicle is assumed to be available where needed).
    - Road network is defined by `(road l1 l2)` facts and is static.

    # Heuristic Initialization
    - Identify all packages and their goal locations from the task goals.
    - Identify all vehicles from initial state/static facts (those with capacity).
    - Build the road network graph from static `(road l1 l2)` facts.
    - Identify all relevant locations from static `(road ...)` facts and initial/goal `(at ...)` facts.
    - Compute all-pairs shortest path distances between all relevant locations using BFS on the road network.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current physical location of every package. A package is either:
       - On the ground at a location `l` (`(at package l)` is true). The physical location is `l`.
       - Inside a vehicle `v` (`(in package v)` is true). The physical location is the location of vehicle `v` (`(at v l_v)` is true).
    2. Identify the current location of every vehicle (`(at vehicle l)` is true).
    3. Initialize the total heuristic cost to 0.
    4. For each package `p` and its goal location `l_goal` (determined during initialization):
       a. Check if the package is already at its goal location on the ground (`(at p l_goal)` is in the state). If yes, the cost for this package is 0; continue to the next package.
       b. If the package is not at its goal, determine its current state (on ground or in vehicle) and its current physical location (`l_p_current`).
       c. If the package is on the ground at `l_p_current` (`l_p_current != l_goal`):
          - It needs to be picked up (1 action).
          - The vehicle needs to drive from `l_p_current` to `l_goal`. The estimated cost is the shortest path distance `dist(l_p_current, l_goal)`.
          - It needs to be dropped off at `l_goal` (1 action).
          - The cost for this package is `1 + dist(l_p_current, l_goal) + 1`.
       d. If the package is inside a vehicle `v` which is at `l_p_current`:
          - It needs to be dropped off at `l_goal` (1 action).
          - If `l_p_current != l_goal`, the vehicle needs to drive from `l_p_current` to `l_goal`. The estimated cost is `dist(l_p_current, l_goal)`.
          - The cost for this package is `1 + dist(l_p_current, l_goal)`. (Note: if `l_p_current == l_goal`, `dist` is 0, cost is 1).
       e. If the shortest path distance `dist(l_p_current, l_goal)` is infinite (no path exists), the problem is likely unsolvable from this state; return infinity.
       f. Add the calculated cost for this package to the total heuristic.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts,
           building the road network, and computing shortest path distances."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Identify packages and their goal locations
        self.goal_locations = {}
        self.packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        # 2. Identify vehicles (objects with capacity)
        self.vehicles = set()
        for fact in initial_state | static_facts:
            parts = get_parts(fact)
            if parts[0] == "capacity":
                vehicle = parts[1]
                self.vehicles.add(vehicle)

        # 3. Build the road network graph and identify all relevant locations
        self.road_graph = {}
        locations_set = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.road_graph.setdefault(l1, set()).add(l2)
                locations_set.add(l1)
                locations_set.add(l2)

        # Also collect locations mentioned in initial/goal states for packages/vehicles
        for fact in initial_state | self.goals:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # The second argument of 'at' is a location
                 locations_set.add(parts[2])

        self.locations = list(locations_set) # Store as list

        # 4. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, dist = q.popleft() # BFS queue

                # Store distance
                self.distances[(start_loc, current_loc)] = dist

                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

        # For locations not reachable, distance query will return None, which we handle in __call__

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Map to store current state of each package: {'at': loc} or {'in': veh}
        package_current_state = {}
        # Map to store current location of each vehicle: veh -> loc
        vehicle_location = {}

        # Populate the maps by iterating through the current state facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_state[obj] = {'at': loc}
                elif obj in self.vehicles:
                    vehicle_location[obj] = loc
            elif parts[0] == 'in':
                pkg, veh = parts[1], parts[2]
                if pkg in self.packages and veh in self.vehicles:
                     package_current_state[pkg] = {'in': veh}

        total_heuristic = 0  # Initialize action cost counter.
        unreachable = False # Flag to indicate if any required path is missing

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal (on the ground)
            if f'(at {package} {goal_location})' in state:
                continue # Package is already at its goal location on the ground.

            # Package is not at goal. Find its current state and physical location.
            current_state_info = package_current_state.get(package)

            if current_state_info is None:
                # Package state unknown - potentially unsolvable or malformed state
                unreachable = True
                break # Exit package loop

            if 'at' in current_state_info:
                # Package is on the ground at current_state_info['at']
                l_current = current_state_info['at']
                # Cost: pick + drive from l_current to goal_location + drop
                # Estimated drive cost is shortest path distance
                dist = self.distances.get((l_current, goal_location))

                if dist is None: # No path exists from current location to goal location
                    unreachable = True
                    break # Exit package loop

                total_heuristic += 1 + dist + 1 # 1 (pick) + drive cost + 1 (drop)

            elif 'in' in current_state_info:
                # Package is in vehicle current_state_info['in']
                veh = current_state_info['in']
                l_v = vehicle_location.get(veh)

                if l_v is None:
                    # Vehicle location unknown - potentially unsolvable or malformed state
                    unreachable = True
                    break # Exit package loop

                # Package's current physical location is the vehicle's location
                l_current_physical = l_v

                # Cost: drop + drive vehicle from l_current_physical to goal_location
                # Estimated drive cost is shortest path distance
                dist = self.distances.get((l_current_physical, goal_location))

                if dist is None: # No path exists from vehicle's location to goal location
                    unreachable = True
                    break # Exit package loop

                total_heuristic += 1 + dist # 1 (drop) + drive cost

            else:
                 # Should not happen based on how package_current_state is built
                 unreachable = True
                 break # Exit package loop


        if unreachable:
            # Return a large value indicating unsolvable or very high cost
            return float('inf')

        return total_heuristic
