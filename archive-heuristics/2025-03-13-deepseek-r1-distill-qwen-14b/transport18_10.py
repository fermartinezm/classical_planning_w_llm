from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their target locations. It considers the vehicle's capacity and the need
    to move between locations efficiently.

    # Assumptions:
    - Vehicles can carry multiple packages, subject to their capacity constraints.
    - Packages must be picked up and dropped off at specific locations.
    - The robot may need to make multiple trips to transport all packages.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Extracts static facts including road connections and vehicle capacities.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine if it is already at its target location. If so, no actions are needed.
    2. For packages not at their target locations:
       a. If the package is in a vehicle, check if the vehicle is at the target location. If not, calculate the actions needed to move the vehicle.
       b. If the package is on the ground, check if a vehicle is present at the package's location. If not, calculate the actions needed to move a vehicle there.
       c. Calculate the actions needed to pick up the package, drive to the target location, and drop off the package.
    3. Sum the actions for all packages, considering the vehicle's capacity to carry multiple packages per trip.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract road connections
        self.roads = set()
        for fact in static_facts:
            if fnmatch(fact, '(road * *)'):
                parts = fact[1:-1].split()
                self.roads.add((parts[1], parts[2]))

        # Extract vehicle capacities and capacity predecessors
        self.capacity = {}
        self.capacity_predecessors = {}
        for fact in static_facts:
            if fnmatch(fact, '(capacity * * *)'):
                _, vehicle, size1, size2 = fact[1:-1].split()
                self.capacity[vehicle] = size2
            elif fnmatch(fact, '(capacity-predecessor * *)'):
                _, s1, s2 = fact[1:-1].split()
                self.capacity_predecessors[s1] = s2

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = goal[1:-1].split()
            if predicate == 'at':
                self.goal_locations[package] = location

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state
        current_locations = {}
        in_vehicle = {}

        # Extract current locations of all objects
        for fact in state:
            if fnmatch(fact, '(at * * *)'):
                _, obj, loc = fact[1:-1].split()
                current_locations[obj] = loc
            elif fnmatch(fact, '(in * * *)'):
                _, package, vehicle = fact[1:-1].split()
                in_vehicle[package] = vehicle
                if vehicle not in current_locations:
                    current_locations[vehicle] = None  # Vehicle's location will be determined separately

        # Get vehicle locations
        vehicle_locations = {}
        for fact in state:
            if fnmatch(fact, '(at * vehicle *)'):
                _, vehicle, loc = fact[1:-1].split()
                vehicle_locations[vehicle] = loc

        total_actions = 0

        # Process each package
        for package in self.goal_locations:
            goal_loc = self.goal_locations[package]
            current_loc = current_locations.get(package, None)
            in_vehicle_flag = package in in_vehicle

            if current_loc == goal_loc:
                continue  # No actions needed

            if in_vehicle_flag:
                vehicle = in_vehicle[package]
                vehicle_loc = vehicle_locations[vehicle]
                if vehicle_loc != goal_loc:
                    # Need to drive to goal location
                    total_actions += 1  # Drive action
            else:
                # Package is on the ground
                if current_loc is None:
                    continue  # Should not happen, but skip if unknown

                # Check if a vehicle is present at the current location
                has_vehicle = any(vehicle_locations[veh] == current_loc for veh in vehicle_locations)
                if not has_vehicle:
                    # Need to drive a vehicle to the current location
                    # Assume closest vehicle (simplified)
                    total_actions += 1  # Drive vehicle to current_loc

                # Actions: pick up, drive to goal, drop off
                total_actions += 3  # pick-up, drive, drop

        return total_actions
