from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transport23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the need to drive vehicles to packages, pick up packages, drive vehicles to goal locations, and drop packages.

    # Assumptions
    - Each package needs to be picked up and dropped off exactly once.
    - Vehicles can carry only one package at a time (simplified capacity).
    - The heuristic ignores capacity constraints and size relationships.
    - The heuristic assumes that there is always a path between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Create a dictionary mapping packages to their goal locations.
    - Create a dictionary mapping vehicles to their current locations.
    - Create a dictionary mapping packages to their current locations.
    - Store the road network information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. For each package, determine its goal location.
    3. If a package is already at its goal location, the cost is 0.
    4. If a package is not at its goal location:
       a. Find the closest vehicle to the package.
       b. Estimate the cost to drive the vehicle to the package's location.
       c. Add the cost of picking up the package.
       d. Estimate the cost to drive the vehicle to the package's goal location.
       e. Add the cost of dropping off the package.
    5. Sum the costs for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.package_goals[package] = location

        # Extract road network information.
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                location1 = parts[1]
                location2 = parts[2]
                self.roads.add((location1, location2))

    def __call__(self, node):
        """
        Estimate the number of actions required to move all packages to their goal locations.
        """
        state = node.state

        # Check if the state is a goal state.
        if all(goal in state for goal in self.goals):
            return 0

        # Extract current locations of packages and vehicles.
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                location = parts[2]
                if any(match(o, obj) for o in ["p*", "package*"]):
                    package_locations[obj] = location
                elif any(match(v, obj) for v in ["v*", "vehicle*"]):
                    vehicle_locations[obj] = location

        total_cost = 0
        for package, current_location in package_locations.items():
            if package in self.package_goals:
                goal_location = self.package_goals[package]
                if current_location != goal_location:
                    # Find the closest vehicle to the package.
                    closest_vehicle = None
                    min_distance = float('inf')
                    for vehicle, vehicle_location in vehicle_locations.items():
                        distance = self.shortest_path(vehicle_location, current_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_vehicle = vehicle

                    # Estimate the cost to drive the vehicle to the package, pick up, drive to goal, and drop off.
                    drive_to_package_cost = self.shortest_path(vehicle_locations[closest_vehicle], current_location)
                    pick_up_cost = 1
                    drive_to_goal_cost = self.shortest_path(current_location, goal_location)
                    drop_off_cost = 1

                    total_cost += drive_to_package_cost + pick_up_cost + drive_to_goal_cost + drop_off_cost

        return total_cost

    def shortest_path(self, start, end):
        """
        Estimate the shortest path between two locations based on the road network.
        This is a simplified estimate and does not guarantee the actual shortest path.
        """
        if start == end:
            return 0

        # Simplified path cost: 1 if directly connected, 2 otherwise.
        if (start, end) in self.roads or (end, start) in self.roads:
            return 1
        else:
            return 2
