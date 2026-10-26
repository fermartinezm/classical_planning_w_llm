from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations. It considers the current locations of packages and vehicles, and calculates the minimal number of drive, pick-up, and drop actions required.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - The goal is to have all packages at their specified target locations.
    - If a package is already at its goal location, no actions are needed for it.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goals.
    - Extracts static facts including vehicle capacities and road connections.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal location for each package.
    2. For each package, determine:
       - If it's already at the goal location (cost: 0).
       - If it's inside a vehicle:
         - Check if the vehicle is at the goal location (cost: 0 if yes).
         - If not, calculate the drive actions needed for the vehicle to reach the goal.
       - If it's on the ground:
         - Find the nearest vehicle location.
         - Calculate drive actions for the vehicle to reach the package.
         - Calculate drive actions for the vehicle to reach the goal location.
    3. Sum the costs for all packages, adjusting for shared vehicle usage.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract vehicle capacities from static facts
        self.vehicle_capacities = {}
        for fact in self.static:
            if fnmatch(fact, '(capacity ?v - vehicle ?s1 - size)'):
                parts = fact[1:-1].split()
                vehicle = parts[1]
                capacity = parts[3]
                self.vehicle_capacities[vehicle] = capacity

        # Extract road connections from static facts
        self.roads = {}
        for fact in self.static:
            if fnmatch(fact, '(road ?l1 ?l2 - location)'):
                l1, l2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            if fnmatch(goal, '(at ?p - package ?l - location)'):
                package = goal[1:-1].split()[1]
                location = goal[1:-1].split()[2]
                self.package_goals[package] = location

    def __call__(self, node):
        """Estimate the minimal number of actions to reach the goal state."""
        state = node.state

        # Extract current locations of packages and vehicles
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            if fnmatch(fact, '(at ?x - package ?l - location)'):
                package = fact[1:-1].split()[1]
                location = fact[1:-1].split()[2]
                package_locations[package] = location
            elif fnmatch(fact, '(at ?v - vehicle ?l - location)'):
                vehicle = fact[1:-1].split()[1]
                location = fact[1:-1].split()[2]
                vehicle_locations[vehicle] = location
            elif fnmatch(fact, '(in ?p - package ?v - vehicle)'):
                vehicle = fact[1:-1].split()[2]
                vehicle_locations[vehicle] = vehicle_locations.get(vehicle, None)

        total_cost = 0

        # For each package, calculate the required actions
        for package, goal_location in self.package_goals.items():
            current_location = package_locations.get(package, None)

            if current_location == goal_location:
                continue  # No cost if already at goal

            # Check if package is in a vehicle
            in_vehicle = package in [fact[1:-1].split()[1] for fact in state if fnmatch(fact, '(in ?p - package ?v - vehicle)')]
            if in_vehicle:
                # Find the vehicle containing the package
                for fact in state:
                    if fnmatch(fact, f'(in {package} ?v - vehicle)'):
                        vehicle = fact[1:-1].split()[2]
                        break
                # Get the vehicle's current location
                vehicle_location = vehicle_locations.get(vehicle, None)
                if vehicle_location == goal_location:
                    continue  # Vehicle is already at goal
                else:
                    # Calculate drive actions needed for vehicle to reach goal
                    # Simplified: assume direct path exists
                    total_cost += 2  # Drive to goal and drop package
            else:
                # Package is on the ground
                # Find nearest vehicle location
                nearest_vehicle = None
                for vehicle, loc in vehicle_locations.items():
                    if loc != current_location:
                        nearest_vehicle = loc
                        break
                if nearest_vehicle is None:
                    # No vehicle available, assume cost is infinite (unsolvable)
                    return float('inf')
                else:
                    # Calculate drive actions: vehicle to package location, then to goal
                    total_cost += 2  # Drive to package and then to goal

        return total_cost
