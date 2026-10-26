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


class transport15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages not at their goal locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic does not consider capacity constraints or optimal routes.
    - It assumes that a vehicle is always available at the package's current location.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. Compare the current location of each package with its goal location.
    3. For each package not at its goal location, estimate the number of actions required:
       - 1 pick-up action
       - Estimate drive actions based on the number of road connections needed to reach the goal location. A simple approach is to assume a minimum of 1 drive action.
       - 1 drop action
    4. Sum the estimated number of actions for all packages to obtain the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Road network information.
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
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        package_locations = {}

        # Extract current locations of packages from the state.
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, location = get_parts(fact)[1], get_parts(fact)[2]
                # Check if the object is a package
                is_package = False
                for package in self.goal_locations.keys():
                    if package == obj:
                        is_package = True
                        break
                if is_package:
                    package_locations[obj] = location

        # Calculate the heuristic value.
        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            if package in package_locations:
                current_location = package_locations[package]
                if current_location != goal_location:
                    # Estimate the number of actions required: pick-up + drive + drop
                    heuristic_value += 3  # 1 pick-up, 1 drive (minimum), 1 drop
            else:
                # Package is not in 'at' predicate, check if it is 'in' a vehicle
                in_vehicle = False
                for fact in state:
                    if match(fact, "in", package, "*"):
                        in_vehicle = True
                        break
                if in_vehicle:
                    heuristic_value += 2 # drop + drive + drop
                else:
                    heuristic_value += 3  # 1 pick-up, 1 drive (minimum), 1 drop

        # If the goal is already reached, return 0.
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        return heuristic_value
