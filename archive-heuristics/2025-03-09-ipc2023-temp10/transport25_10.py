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


class transport25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic ignores capacity constraints and assumes vehicles can carry all packages at once.
    - The heuristic assumes that vehicles are always at a location where they can pick up packages.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network from the static facts to estimate driving distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. For each package, check if it is at its goal location.
    3. If a package is not at its goal location, estimate the number of actions required to move it:
       - One pick-up action.
       - Estimate the number of drive actions to reach the goal location.
       - One drop action.
    4. Sum the estimated number of actions for all packages not at their goal locations.
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
        Estimate the number of actions required to reach the goal state from the given state.
        """
        state = node.state

        # Extract current locations of packages.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and args[0] in self.goal_locations:
                package, location = args
                current_locations[package] = location

        # Calculate the heuristic value.
        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations or current_locations[package] != goal_location:
                heuristic_value += 3  # pick-up + drive + drop

        return heuristic_value
