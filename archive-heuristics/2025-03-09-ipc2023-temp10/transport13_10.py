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


class transport13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to transport all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of pick-up,
    drop, and drive actions needed to move them. It also takes into account the current location of vehicles
    and packages.

    # Assumptions
    - Each package needs to be picked up, transported by a vehicle, and dropped at its destination.
    - The heuristic assumes that there is always a vehicle available to transport a package.
    - The heuristic does not consider capacity constraints.
    - The heuristic assumes that the vehicle can directly drive to the package and the goal location.

    # Heuristic Initialization
    - Extract the goal locations for each package.
    - Store the road network information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. Extract the goal locations of all packages from the task goals.
    3. For each package, check if it is at its goal location.
    4. If a package is not at its goal location, estimate the number of actions required to move it to its goal:
       - One pick-up action to load the package into a vehicle.
       - One drive action to move the vehicle to the goal location.
       - One drop action to unload the package at the goal location.
    5. Sum the estimated number of actions for all packages.
    6. If all packages are at their goal locations, return 0.
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
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the given state.
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

        # Calculate the heuristic value.
        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            if package in package_locations:
                current_location = package_locations[package]
                if current_location != goal_location:
                    heuristic_value += 3  # pick-up + drive + drop
            else:
                # Package is in a vehicle, need to account for that
                heuristic_value += 3

        # If all packages are at their goal locations, return 0.
        if heuristic_value == 0:
            return 0

        return heuristic_value
