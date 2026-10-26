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


class transport21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of
    pick-up, drop, and drive actions needed to transport them.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic does not take into account vehicle capacity or size constraints.
    - The heuristic assumes that there are always roads between locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the goal locations for each package from the task goals.
    2. For each package, check if it is at its goal location in the current state.
    3. If a package is not at its goal location, estimate the number of actions needed to transport it:
       - One pick-up action to load the package into a vehicle.
       - One drive action to move the vehicle to the goal location.
       - One drop action to unload the package at the goal location.
    4. Sum the estimated number of actions for all packages that are not at their goal locations.
    5. If all packages are at their goal locations, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal locations for each package.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                self.package_goals[parts[1]] = parts[2]

    def __call__(self, node):
        """
        Estimate the number of actions required to move all packages to their goal locations.
        """
        state = node.state
        cost = 0

        for package, goal_location in self.package_goals.items():
            package_at_goal = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == package and parts[2] == goal_location:
                    package_at_goal = True
                    break

            if not package_at_goal:
                # Estimate cost: pick-up + drive + drop
                cost += 3

        return cost
