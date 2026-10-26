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


class transport20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic does not consider capacity constraints or size limitations of vehicles.
    - It assumes that there are enough vehicles at the right locations to pick up the packages.
    - The heuristic only counts the number of packages not at their goal locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Store the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the current state.
    2. Compare the current location of each package with its goal location.
    3. Count the number of packages that are not at their goal locations.
    4. For each package not at its goal, estimate the number of actions required:
       - 1 pick-up action
       - 1 drop action
       - Number of drive actions is estimated as 1 (simplified).
    5. Sum up the estimated number of actions for all misplaced packages.
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
                l1, l2 = get_parts(fact)[1:]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state
        misplaced_packages = 0

        # Extract current locations of packages.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and args[0] in self.goal_locations:
                package, location = args
                current_locations[package] = location

        # Count misplaced packages.
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations or current_locations[package] != goal_location:
                misplaced_packages += 1

        # Estimate the number of actions.
        if misplaced_packages == 0:
            return 0
        else:
            # Each misplaced package requires at least a pick-up, a drop, and some driving.
            return misplaced_packages * 3
