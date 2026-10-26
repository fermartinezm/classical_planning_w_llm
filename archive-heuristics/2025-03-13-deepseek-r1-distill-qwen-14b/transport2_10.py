from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations. It considers the current locations of packages and vehicles, and calculates the required pick-up, drop, and drive actions.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each pick-up or drop action counts as one step.
    - Driving between two connected locations counts as one step.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goals.
    - Stores static facts including road connections and vehicle capacities.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it's inside a vehicle.
    2. If the package is already at its goal location, no actions are needed.
    3. If the package is inside a vehicle:
       a. Determine the vehicle's current location.
       b. Calculate the number of drive actions needed to move the vehicle to the package's goal location.
       c. Add one drop action to unload the package.
    4. If the package is on the ground:
       a. Calculate the number of drive actions needed to move a vehicle to the package's current location.
       b. Add one pick-up action.
       c. Calculate the number of drive actions needed to move the vehicle to the package's goal location.
       d. Add one drop action.
    5. Sum all the required actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at':
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

        # Preprocess static facts: road connections and vehicle capacities
        self.roads = {}
        for fact in self.static:
            if fact.startswith('(road'):
                loc1, loc2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if loc1 not in self.roads:
                    self.roads[loc1] = []
                self.roads[loc1].append(loc2)
                if loc2 not in self.roads:
                    self.roads[loc2] = []
                self.roads[loc2].append(loc1)

        self.capacity = {}
        for fact in self.static:
            if fact.startswith('(capacity'):
                vehicle = fact[1:-1].split()[1]
                size = fact[1:-1].split()[2]
                self.capacity[vehicle] = size

    def __call__(self, node):
        """Estimate the minimum number of actions needed to reach the goal state."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Track locations of packages and vehicles
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == 'at':
                obj, loc = args
                current_locations[obj] = loc
            elif predicate == 'in':
                package, vehicle = args
                current_locations[package] = vehicle

        total_actions = 0

        # For each package, calculate required actions
        for package, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(package, None)
            if current_loc == goal_loc:
                continue  # Package is already at goal

            if current_loc in self.capacity:
                # Package is inside a vehicle
                vehicle = current_loc
                vehicle_loc = None
                for fact in state:
                    if match(fact, 'at', vehicle, '*'):
                        vehicle_loc = fact[1:-1].split()[2]
                        break
                if vehicle_loc is None:
                    continue  # Vehicle not found

                # Calculate drive actions from vehicle's current location to package's goal
                path_length = self._get_path_length(vehicle_loc, goal_loc)
                total_actions += path_length + 1  # drive + drop
            else:
                # Package is on the ground
                # Need to move a vehicle to package's location, then to goal
                # Find any vehicle that can carry the package
                vehicle = None
                for fact in state:
                    if match(fact, 'at', '*', current_loc) and fact[1:-1].split()[0] == 'at':
                        vehicle = fact[1:-1].split()[1]
                        break

                if vehicle is None:
                    continue  # No vehicle available

                # Move vehicle to package's location
                path_length_to_package = self._get_path_length(vehicle, current_loc)
                total_actions += path_length_to_package + 1  # drive + pick-up

                # Move vehicle to package's goal location
                path_length_to_goal = self._get_path_length(current_loc, goal_loc)
                total_actions += path_length_to_goal + 1  # drive + drop

        return total_actions

    def _get_path_length(self, start, end):
        """Estimate the shortest path length between two locations using BFS."""
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, steps = queue.pop(0)
            if current == end:
                return steps
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.roads.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, steps + 1))
        return float('inf')  # No path found
