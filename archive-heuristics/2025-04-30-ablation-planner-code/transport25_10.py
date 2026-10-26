from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential invalid format defensively
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
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
    Estimates the total number of actions (pick-up, drop, drive) required to move all packages to their goal locations, ignoring vehicle capacity constraints and vehicle availability conflicts. It sums the estimated cost for each package independently.

    # Assumptions
    - The road network is bidirectional.
    - Any vehicle can pick up any package (capacity is ignored).
    - A suitable vehicle is always available at the required location when needed for a pick-up or drop.
    - The cost of a `drive` action is 1 per road segment traversed.
    - The cost of `pick-up` and `drop` actions is 1.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph representation of the road network from static `road` facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies vehicle and package object names from the task definition for state parsing.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state by verifying if all goal facts are present in the state. If yes, return 0.
    2. Parse the current state to determine the current location of each package (either `at` a location or `in` a vehicle) and the current location of each vehicle.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each package that is not yet at its goal location:
       a. If the package is already at its goal location (checked by the presence of the specific `(at package_name goal_loc)` fact in the state), its cost contribution is 0. Continue to the next package.
       b. If the package is not at its goal location, determine its current status (on the ground or in a vehicle).
       c. If the package is currently on the ground at `current_loc`:
          - The estimated cost for this package includes: 1 (for `pick-up`) + shortest_path_distance(`current_loc`, `goal_loc`) (for `drive` actions) + 1 (for `drop`). Add this to the total cost.
       d. If the package is currently inside a vehicle `v`, and the vehicle is at `vehicle_loc`:
          - The estimated cost for this package includes: shortest_path_distance(`vehicle_loc`, `goal_loc`) (for `drive` actions, only if `vehicle_loc` is not `goal_loc`) + 1 (for `drop`). Add this to the total cost.
       e. If the package's status cannot be determined from the state (e.g., not mentioned in `at` or `in` facts, or in an unknown type of object), return infinity to indicate a potentially invalid or unsolvable state from this point.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing shortest paths."""
        self.goals = task.goals  # Goal conditions (list of fact strings)

        # Extract object types from the task definition
        self.vehicle_names = {obj.name for obj in task.objects if obj.type == 'vehicle'}
        self.package_names = {obj.name for obj in task.objects if obj.type == 'package'}
        locations = [obj.name for obj in task.objects if obj.type == 'location']

        # Build the road graph (adjacency list)
        adj = {loc: [] for loc in locations}
        for fact in task.static:
            predicate, *args = get_parts(fact)
            if predicate == "road" and len(args) == 2:
                l1, l2 = args
                # Ensure locations are valid nodes in our graph
                if l1 in adj and l2 in adj:
                    adj[l1].append(l2)
                    adj[l2].append(l1) # Assuming roads are bidirectional

        # Compute all-pairs shortest paths using BFS from each location
        self.dist = {}
        for start_loc in locations:
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                current_loc, d = queue.popleft()
                self.dist[(start_loc, current_loc)] = d

                for neighbor in adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        # Fill in unreachable distances with infinity
        for l1 in locations:
            for l2 in locations:
                if (l1, l2) not in self.dist:
                    self.dist[(l1, l2)] = float('inf')

        # Extract package goals from the task goals
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                package, location = args
                # Only store goals for packages we know about
                if package in self.package_names:
                    self.package_goals[package] = location
                # else: print(f"Warning: Goal for non-package object {package} ignored.")
            # else: print(f"Warning: Non-'at' goal fact '{goal}' ignored by heuristic.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # Check if it's a goal state by verifying all goal facts are present
        if all(goal_fact in state for goal_fact in self.goals):
            return 0

        # If not a goal state, calculate the heuristic
        package_locations = {} # Map package name to its current status (location name or vehicle name)
        vehicle_locations = {} # Map vehicle name to its current location name

        # Parse the state to find locations of packages and vehicles
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and len(args) == 2:
                obj_name, loc_name = args
                if obj_name in self.vehicle_names:
                    vehicle_locations[obj_name] = loc_name
                elif obj_name in self.package_names:
                    package_locations[obj_name] = loc_name
            elif predicate == "in" and len(args) == 2:
                package_name, vehicle_name = args
                if package_name in self.package_names and vehicle_name in self.vehicle_names:
                     package_locations[package_name] = vehicle_name # Store vehicle name as package's status
                # else: print(f"Warning: 'in' fact with non-package/vehicle objects: {fact}")


        total_cost = 0

        # Calculate cost for each package that is not yet at its goal
        for package_name, goal_loc in self.package_goals.items():
            # Check if the specific goal fact for this package is met
            goal_fact_for_package = f"(at {package_name} {goal_loc})"
            if goal_fact_for_package in state:
                continue # This package is already at its goal location

            # If the package is not at its goal location, estimate cost
            current_status = package_locations.get(package_name)

            if current_status is None:
                # Package not found in state facts - should not happen in valid states
                # print(f"Warning: Package {package_name} not found in state facts.")
                return float('inf') # Indicate unsolvable or invalid state

            if current_status in self.vehicle_names: # Package is in a vehicle
                vehicle_name = current_status
                vehicle_loc = vehicle_locations.get(vehicle_name)

                if vehicle_loc is None:
                     # Vehicle location unknown - should not happen in valid states
                     # print(f"Warning: Vehicle {vehicle_name} carrying {package_name} has no location.")
                     return float('inf') # Indicate unsolvable or invalid state

                # Cost: Drive vehicle from vehicle_loc to goal_loc + Drop
                # Drive cost is 0 if vehicle is already at goal_loc
                drive_cost = self.dist.get((vehicle_loc, goal_loc), float('inf'))
                if drive_cost == float('inf'):
                    # Unreachable goal location for the vehicle
                    # print(f"Warning: Goal location {goal_loc} unreachable from vehicle location {vehicle_loc} for package {package_name}.")
                    return float('inf') # Indicate unsolvable path

                total_cost += drive_cost
                total_cost += 1 # Drop action

            elif current_status in self.dist: # Package is at a location (check if it's a known location node in the graph)
                current_loc = current_status

                # Cost: Pick up + Drive from current_loc to goal_loc + Drop
                drive_cost = self.dist.get((current_loc, goal_loc), float('inf'))
                if drive_cost == float('inf'):
                    # Unreachable goal location
                    # print(f"Warning: Goal location {goal_loc} unreachable from package location {current_loc} for package {package_name}.")
                    return float('inf') # Indicate unsolvable path

                total_cost += 1 # Pick up action
                total_cost += drive_cost
                total_cost += 1 # Drop action
            else:
                # Package status is neither a vehicle name nor a known location name
                # This indicates an issue with state representation or parsing.
                # Assign a large cost.
                # print(f"Warning: Package {package_name} has unexpected status '{current_status}'.")
                return float('inf') # Indicate unsolvable or invalid state

        return total_cost
