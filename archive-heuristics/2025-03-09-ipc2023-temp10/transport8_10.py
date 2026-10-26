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


class transport8Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    to their respective goal locations. It considers the number of packages at
    incorrect locations and estimates the number of pick-up, drop, and drive
    actions required.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its
      destination.
    - The heuristic does not explicitly model capacity constraints or size
      relationships.
    - The heuristic assumes that vehicles are always available at the package's
      current location.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. For each package, check if it is at its goal location.
    3. If a package is not at its goal location, estimate the number of actions
       required to move it to its goal location. This includes:
        - One pick-up action.
        - One drop action.
        - At least one drive action (estimated as 1).
    4. Sum the estimated number of actions for all packages to get the total
       heuristic value.
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
                self.roads.add((l2, l1))

    def __call__(self, node):
        """
        Estimate the number of actions needed to reach the goal state from the
        given state.
        """
        state = node.state
        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, location = get_parts(fact)[1:]
                # Check if the object is a package
                is_package = False
                for package in self.goal_locations.keys():
                    if package == obj:
                        is_package = True
                        break
                if is_package:
                    package_locations[obj] = location

        heuristic_value = 0
        for package, current_location in package_locations.items():
            goal_location = self.goal_locations[package]
            if current_location != goal_location:
                # Estimate cost: pick-up + drive + drop
                heuristic_value += 3

        return heuristic_value
