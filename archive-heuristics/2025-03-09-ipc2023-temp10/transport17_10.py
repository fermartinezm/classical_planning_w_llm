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


class transport17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic ignores capacity constraints and assumes vehicles can carry all packages simultaneously.
    - The heuristic assumes that there is always a path between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Extract the road network from the static facts to estimate driving costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Information:
       - Identify the goal location for each package.

    2. Identify Packages at Incorrect Locations:
       - Determine the current location of each package.
       - Compare the current location with the goal location.
       - Count the number of packages that are not at their goal locations.

    3. Estimate the Number of Actions:
       - For each package at the wrong location, estimate the number of actions required:
         - 1 pick-up action.
         - 1 drop action.
         - Estimate the number of drive actions. This is simplified to 1 drive action per package.

    4. Calculate Total Cost:
       - Sum the estimated costs for all packages.
       - The total cost is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Extract road network information (not used in this simplified heuristic).
        self.roads = set()
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state.
        """
        state = node.state
        packages_at_wrong_location = 0

        # Determine the current location of each package.
        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                package, location = get_parts(fact)[1], get_parts(fact)[2]
                package_locations[package] = location

        # Count the number of packages at the wrong location.
        for package, goal_location in self.goal_locations.items():
            if package not in package_locations or package_locations[package] != goal_location:
                packages_at_wrong_location += 1

        # Estimate the number of actions required (pick-up, drop, drive).
        heuristic_value = 0
        if packages_at_wrong_location > 0:
            heuristic_value = packages_at_wrong_location * 3  # pick-up, drop, and drive

        return heuristic_value
