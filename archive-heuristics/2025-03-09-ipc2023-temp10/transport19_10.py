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


class transport19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic does not consider capacity constraints or optimal routes.
    - It assumes that each package requires at least one pick-up, one drop, and some number of drive actions.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road network information from the static facts to estimate driving costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. For each package, check if it is at its goal location.
    3. If a package is not at its goal location, estimate the number of actions required:
        - One pick-up action to load the package onto a vehicle.
        - One drop action to unload the package at its goal location.
        - Estimate the number of drive actions required to move the vehicle from the package's current location to its goal location. This is a simplified estimate and does not consider the shortest path. We simply add 1 if the vehicle and package are not at the same location.
    4. Sum the estimated number of actions for all packages to get the heuristic value.
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
                l1, l2 = get_parts(fact)[1:]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current locations of packages.
        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                package, location = get_parts(fact)[1:]
                if package in self.goal_locations:
                    package_locations[package] = location

        # Calculate the heuristic value.
        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            if package not in package_locations or package_locations[package] != goal_location:
                heuristic_value += 2  # pick-up and drop

                # Add a drive action if the package and a vehicle are not at the same location.
                vehicle_at_package_location = False
                for fact in state:
                    if match(fact, "at", "*", package_locations.get(package, '')):
                        vehicle = get_parts(fact)[1]
                        if vehicle.startswith('v'):
                            vehicle_at_package_location = True
                            break
                if not vehicle_at_package_location:
                    heuristic_value += 1

        return heuristic_value
