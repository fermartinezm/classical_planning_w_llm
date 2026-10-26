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


class transport11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to transport all packages to their goal locations.
    It considers the number of packages at incorrect locations and the number of drive actions needed to bring vehicles to the packages and then to the goal locations.

    # Assumptions
    - Each package needs to be picked up and dropped at least once.
    - Vehicles need to drive to the packages and to the goal locations.
    - The heuristic does not explicitly consider capacity constraints.

    # Heuristic Initialization
    - Extract the goal locations for each package.
    - Extract the road network to estimate driving costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. Extract the goal locations of all packages from the task goals.
    3. For each package, check if it is at its goal location.
    4. If a package is not at its goal location, estimate the number of actions required to transport it:
       - One pick-up action.
       - One drop action.
       - Estimate the number of drive actions:
         - Find a vehicle at the same location as the package. If no such vehicle exists, estimate the number of drive actions to bring a vehicle to the package's location.
         - Estimate the number of drive actions to bring the vehicle from the package's location to the goal location.
    5. Sum up the estimated number of actions for all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network information.
        """
        self.goals = task.goals

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Extract road network information.
        self.roads = set()
        for fact in task.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current locations of packages and vehicles.
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                entity, location = args
                if entity in self.goal_locations:  # It's a package
                    package_locations[entity] = location
                else:  # It's a vehicle
                    vehicle_locations[entity] = location

        # Initialize the heuristic value.
        heuristic_value = 0

        # For each package, estimate the cost to reach its goal location.
        for package, goal_location in self.goal_locations.items():
            if package in package_locations:
                current_location = package_locations[package]

                # If the package is not at its goal location, estimate the cost.
                if current_location != goal_location:
                    # Add cost for pick-up and drop actions.
                    heuristic_value += 2

                    # Estimate the cost for drive actions.
                    # First, check if there is a vehicle at the same location as the package.
                    vehicle_at_package = False
                    for vehicle, location in vehicle_locations.items():
                        if location == current_location:
                            vehicle_at_package = True
                            break

                    # If no vehicle is at the package location, estimate the cost to bring one there.
                    if not vehicle_at_package:
                        heuristic_value += 1  # Assume at least one drive action is needed

                    # Estimate the cost to drive the vehicle from the package location to the goal location.
                    heuristic_value += 1  # Assume at least one drive action is needed

        # Return the estimated heuristic value.
        return heuristic_value
