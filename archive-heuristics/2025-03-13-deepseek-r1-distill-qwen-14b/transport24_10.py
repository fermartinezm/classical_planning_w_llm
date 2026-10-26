from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their goal locations using vehicles.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each pick-up or drop action counts as one step.
    - Driving between two connected locations counts as one step.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goals.
    - Uses static facts to determine vehicle capacities and road connections.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract goal locations for each package from the task's goals.
    2. For each package, determine its current location and whether it's inside a vehicle.
    3. If the package is already at its goal location, no actions are needed.
    4. If the package is inside a vehicle, check if the vehicle can be driven directly to the goal location.
    5. If the package is on the ground, calculate the number of actions needed to pick it up, drive it to the goal location, and drop it.
    6. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal locations and static facts."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract vehicle capacities from static facts
        self.vehicle_capacities = {}
        for fact in static_facts:
            if fnmatch(fact, '(capacity * * *)'):
                _, vehicle, capacity = fact[1:-1].split()
                self.vehicle_capacities[vehicle] = capacity

        # Extract road connections from static facts
        self.roads = {}
        for fact in static_facts:
            if fnmatch(fact, '(road * * *)'):
                loc1, loc2 = fact[1:-1].split()
                if loc1 not in self.roads:
                    self.roads[loc1] = []
                self.roads[loc1].append(loc2)
                if loc2 not in self.roads:
                    self.roads[loc2] = []
                self.roads[loc2].append(loc1)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if fnmatch(goal, '(at * *)'):
                package, location = goal[1:-1].split()
                self.goal_locations[package] = location

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current locations of packages and vehicles
        current_locations = {}
        for fact in state:
            if fnmatch(fact, '(at * *)'):
                obj, loc = fact[1:-1].split()
                current_locations[obj] = loc
            elif fnmatch(fact, '(in * * *)'):
                package, vehicle = fact[1:-1].split()
                if vehicle in self.vehicle_capacities:
                    current_locations[package] = vehicle

        total_actions = 0

        # For each package, calculate the required actions
        for package, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(package, None)

            if current_loc == goal_loc:
                continue  # Package is already at goal

            if current_loc is None:
                continue  # Package not present in state (should not happen)

            # If the package is in a vehicle, check if the vehicle can be driven to the goal
            if current_loc in self.vehicle_capacities:
                vehicle = current_loc
                vehicle_loc = None
                for fact in state:
                    if fnmatch(fact, '(at * * *)') and fact[1:-1].split()[1] == vehicle:
                        vehicle_loc = fact[1:-1].split()[2]
                        break
                if vehicle_loc is None:
                    continue  # Vehicle not located (should not happen)

                # Check if vehicle can drive directly to goal location
                if vehicle_loc == goal_loc:
                    continue  # Vehicle is already at goal location
                else:
                    # Need to drive the vehicle to the goal location
                    total_actions += 1  # Drive action
            else:
                # Package is on the ground, need to pick it up and drop it
                # Find a vehicle to carry the package
                nearest_vehicle = None
                for fact in state:
                    if fnmatch(fact, '(at * * *)') and fact[1:-1].split()[1] in self.vehicle_capacities:
                        vehicle_loc = fact[1:-1].split()[2]
                        if vehicle_loc == current_loc:
                            nearest_vehicle = fact[1:-1].split()[1]
                            break
                if nearest_vehicle is None:
                    continue  # No vehicle available at current location (should not happen)

                # Drive to the goal location
                total_actions += 1  # Drive action
                # Drop the package
                total_actions += 1  # Drop action

        return total_actions
