from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their respective goal locations using vehicles.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between locations.
    - Each vehicle has a specific capacity that determines how many packages it can carry.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Extracts static facts including road connections and vehicle capacities.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each package, check if it is already at its goal location. If yes, no actions are needed.
    2. If the package is not at its goal:
       a. Determine if the package is inside a vehicle or on the ground.
       b. If inside a vehicle, find the vehicle's current location.
       c. Calculate the number of actions required to move the vehicle to the package's location (if needed), pick up the package, move to the goal location, and drop the package.
       d. If the package is on the ground, calculate the actions needed to pick it up and transport it to the goal.
    3. Sum the actions required for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with task information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract road connections from static facts
        self.roads = set()
        for fact in self.static:
            if fnmatch(fact, '(road * *)'):
                parts = fact[1:-1].split()
                self.roads.add((parts[1], parts[2]))
                self.roads.add((parts[2], parts[1]))

        # Extract capacity information
        self.capacity = {}
        for fact in self.static:
            if fnmatch(fact, '(capacity * * *)'):
                parts = fact[1:-1].split()
                vehicle, size1, size2 = parts[1], parts[2], parts[3]
                self.capacity[vehicle] = (size1, size2)

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if fnmatch(goal, '(at * *)'):
                package, loc = goal[1:-1].split()[1], goal[1:-1].split()[2]
                self.goal_locations[package] = loc

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state
        current_locations = {}
        in_vehicle = {}

        # Extract current locations of all objects
        for fact in state:
            if fnmatch(fact, '(at * *)'):
                obj, loc = fact[1:-1].split()[1], fact[1:-1].split()[2]
                current_locations[obj] = loc
            elif fnmatch(fact, '(in * *)'):
                package, vehicle = fact[1:-1].split()[1], fact[1:-1].split()[2]
                in_vehicle[package] = vehicle

        total_actions = 0

        # For each package, determine the required actions
        for package, goal_loc in self.goal_locations.items():
            if package in current_locations and current_locations[package] == goal_loc:
                continue  # Package is already at goal

            if package in in_vehicle:
                vehicle = in_vehicle[package]
                vehicle_loc = current_locations[vehicle]
            else:
                vehicle = None
                vehicle_loc = current_locations.get(package, None)

            if vehicle:
                # Package is inside a vehicle; need to move the vehicle to the goal
                if vehicle_loc != goal_loc:
                    total_actions += 2  # Drive to goal and drop package
            else:
                # Package is on the ground; need to pick up and transport
                if vehicle_loc is not None and vehicle_loc != goal_loc:
                    total_actions += 2  # Drive to package and pick up, then drive to goal and drop
                else:
                    total_actions += 2  # Pick up and drop

        return total_actions
