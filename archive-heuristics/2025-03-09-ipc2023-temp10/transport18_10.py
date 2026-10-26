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


class transport18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages not at their goal locations and estimates the number of pick-up,
    drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic ignores capacity constraints and assumes vehicles are always available.
    - It assumes that there is always a road between the current location of a vehicle and the location of a package.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. Compare the current location of each package with its goal location.
    3. For each package not at its goal location, estimate the number of actions required:
       - One pick-up action to load the package onto a vehicle.
       - One drop action to unload the package at the goal location.
       - Estimate the number of drive actions required to move the vehicle to the goal location.
         This is simplified by assuming a direct road exists between the current vehicle location and the package's goal location.
    4. Sum the estimated number of actions for all packages to obtain the heuristic value.
    5. If the state is a goal state, return 0.
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

        # If the state is a goal state, return 0.
        if task.goal_reached(state):
            return 0

        # Extract current locations of packages.
        package_locations = {}
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
        for package, current_location in package_locations.items():
            if current_location != self.goal_locations[package]:
                # Estimate the number of actions required: pick-up, drive, drop.
                heuristic_value += 3  # pick-up + drive + drop

        return heuristic_value
