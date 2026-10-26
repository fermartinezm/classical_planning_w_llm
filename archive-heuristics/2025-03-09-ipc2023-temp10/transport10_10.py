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


class transport10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to transport all packages to their goal locations.
    It considers the need to move vehicles to packages, pick up packages, move vehicles to goal locations, and drop packages.

    # Assumptions
    - Each package needs to be picked up by a vehicle, transported to its destination, and dropped.
    - Vehicles need to travel to the initial location of the package and from the package's initial location to the goal location.
    - The heuristic does not consider capacity constraints or size limitations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Extract the road network from the static facts to estimate travel costs between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. For each package, determine its goal location.
    3. For each package, find the closest vehicle to its current location.
    4. Estimate the cost to move the closest vehicle to the package's current location (number of road actions).
    5. Add the cost of picking up the package.
    6. Estimate the cost to move the vehicle from the package's current location to its goal location (number of road actions).
    7. Add the cost of dropping the package.
    8. Sum the costs for all packages to obtain the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Extract road network information.
        self.road_network = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.road_network.add((l1, l2))

    def __call__(self, node):
        """
        Estimate the number of actions required to transport all packages to their goal locations.
        """
        state = node.state
        package_locations = {}
        vehicle_locations = {}

        # Extract current locations of packages and vehicles.
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                if obj in self.goal_locations:  # It's a package
                    package_locations[obj] = location
                else:  # It's a vehicle
                    vehicle_locations[obj] = location

        total_cost = 0
        for package, goal_location in self.goal_locations.items():
            if package in package_locations:
                current_location = package_locations[package]

                # Find the closest vehicle.  For simplicity, assume all vehicles are equally close.
                closest_vehicle = next(iter(vehicle_locations))
                vehicle_location = vehicle_locations[closest_vehicle]

                # Estimate cost to move vehicle to package. For simplicity, assume cost is always 1.
                vehicle_move_cost = 1

                # Cost to pick up the package.
                pickup_cost = 1

                # Estimate cost to move vehicle to goal location. For simplicity, assume cost is always 1.
                goal_move_cost = 1

                # Cost to drop the package.
                drop_cost = 1

                total_cost += vehicle_move_cost + pickup_cost + goal_move_cost + drop_cost
            else:
                # Package is already in a vehicle, assume 1 action to drop it at the goal.
                total_cost += 1

        # If the goal is reached, the heuristic value is 0.
        if node.state >= self.goals:
            return 0

        return total_cost
