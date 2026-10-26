# Import necessary modules
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Utility functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-dependent heuristic for the Transport domain
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated costs for each package
    independently, ignoring vehicle capacity constraints and potential
    multi-package trips.

    # Assumptions
    - The primary goal is to move packages to specific locations.
    - Vehicle capacity and availability are not bottlenecks (optimistic assumption).
    - Roads are bidirectional.
    - The cost of any action (drive, pick-up, drop) is 1.

    # Heuristic Initialization
    - Parses static facts to build the road network graph and precompute
      shortest path distances between all pairs of locations using BFS.
    - Parses goal conditions to identify the target location for each package.
    - Identifies packages and vehicles from initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic is 0.
    2. For each package that is not yet at its goal location:
       a. Determine the package's current status:
          - Is it on the ground at some location `l_current`?
          - Is it inside a vehicle `v`? If so, find the vehicle's location `l_v`.
       b. Estimate the minimum actions required for this package to reach its goal `l_goal`:
          - If on the ground at `l_current`:
            - It needs to be picked up (1 action).
            - It needs to be transported from `l_current` to `l_goal`. The minimum drive actions is the shortest path distance `distance(l_current, l_goal)`.
            - It needs to be dropped at `l_goal` (1 action).
            - Total estimated cost for this package: 1 + distance(l_current, l_goal) + 1.
          - If inside vehicle `v` at `l_v`:
            - It needs to be transported from `l_v` to `l_goal`. The minimum drive actions is the shortest path distance `distance(l_v, l_goal)`.
            - It needs to be dropped at `l_goal` (1 action).
            - Total estimated cost for this package: distance(l_v, l_goal) + 1.
       c. If the goal location is unreachable from the package's current location or vehicle's location, the cost for this package is infinity.
    3. The total heuristic value is the sum of the estimated costs for all packages not at their goal. If any package's goal is unreachable, the total heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Need initial state to identify vehicles

        # 1. Build Road Network and Precompute Distances
        self.locations = set() # Store all locations
        roads = {} # adjacency list: location -> list of connected locations

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                roads.setdefault(l1, []).append(l2)
                roads.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        self.distances = {} # (start_loc, end_loc) -> distance
        for start_loc in self.locations:
            # Perform BFS from start_loc to find distances to all other locations
            queue = deque([(start_loc, 0)])
            visited = {start_loc: 0}
            self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in roads:
                    for neighbor in roads[current_loc]:
                        if neighbor not in visited:
                            visited[neighbor] = dist + 1
                            self.distances[(start_loc, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

        # 2. Identify Packages and Vehicles, and Store Goal Locations
        self.packages = set()
        self.vehicles = set()
        self.goal_locations = {} # package -> goal_location

        # Identify packages from goal facts
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                if len(args) == 2:
                    package, location = args
                    self.goal_locations[package] = location
                    self.packages.add(package)

        # Identify vehicles from initial state capacity facts
        for fact in initial_state_facts:
            predicate, *args = get_parts(fact)
            if predicate == "capacity":
                 if len(args) >= 1: # Capacity takes vehicle and size
                    vehicle = args[0]
                    self.vehicles.add(vehicle)

        # Note: Capacity sizes are not needed for this heuristic calculation.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        task = node.task # Access task from node to check goal_reached

        # If the goal is already reached, the heuristic is 0.
        if task.goal_reached(state):
            return 0

        total_cost = 0
        unreachable = False # Flag to indicate if any package goal is unreachable

        # Map current locations/vehicles for packages and vehicles in the current state
        current_package_location_or_vehicle = {} # package -> location or vehicle name
        current_vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == "at":
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.packages:
                        current_package_location_or_vehicle[obj] = loc
                    elif obj in self.vehicles:
                        current_vehicle_locations[obj] = loc
            elif predicate == "in":
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    if package in self.packages and vehicle in self.vehicles:
                         current_package_location_or_vehicle[package] = vehicle

        # Calculate cost for each package not at its goal
        for package, l_goal in self.goal_locations.items():
            # Check if package is already at goal location by checking the goal fact string
            goal_fact_str = f"(at {package} {l_goal})"
            if goal_fact_str in state:
                 continue # Package is done

            # Package is not at goal. Estimate cost.
            # Find its current location or vehicle.
            current_pos = current_package_location_or_vehicle.get(package)

            if current_pos is None:
                 # Package exists (in goals) but is not 'at' any location and not 'in' any vehicle in the state.
                 # This indicates an invalid or unreachable state.
                 unreachable = True
                 break # No need to check other packages

            # If package is on the ground at current_pos (which is a location)
            if current_pos in self.locations: # Check if current_pos is a known location
                l_current = current_pos
                # Cost: pick-up (1) + drive (distance) + drop (1)
                drive_cost = self.distances.get((l_current, l_goal))
                if drive_cost is None: # No path found
                     unreachable = True
                     break
                total_cost += 1 + drive_cost + 1

            # If package is in a vehicle (current_pos is a vehicle name)
            elif current_pos in self.vehicles:
                v = current_pos
                l_v = current_vehicle_locations.get(v)
                if l_v is None:
                     # Vehicle location not found? Invalid state.
                     unreachable = True
                     break

                # Cost: drive (distance) + drop (1)
                drive_cost = self.distances.get((l_v, l_goal))
                if drive_cost is None: # No path found
                     unreachable = True
                     break
                total_cost += drive_cost + 1

            else:
                # current_pos is neither a known location nor a known vehicle. Invalid state.
                unreachable = True
                break


        if unreachable:
            return float('inf')

        return total_cost
