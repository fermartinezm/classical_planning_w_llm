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


class transport3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages not at their goal locations and estimates the number of drive, pick-up, and drop actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic does not account for vehicle capacity or size constraints.
    - The heuristic assumes that a vehicle is always available at the package's current location.
    - The heuristic does not consider the need to move vehicles between locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Information:
       - From the task goals, determine the destination location for each package.

    2. Identify Packages Not at Goal:
       - Iterate through the current state and identify packages that are not at their intended goal locations.

    3. Estimate Actions for Each Package:
       - For each package not at its goal, estimate the number of actions required:
         - 1 pick-up action to load the package onto a vehicle.
         - 1 drop action to unload the package at the destination.
         - 1 drive action to move the vehicle to the destination.

    4. Aggregate Action Estimates:
       - Sum the estimated number of actions for all packages not at their goal locations.

    5. Return Total Estimated Actions:
       - The total estimated actions represent the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                self.package_goals[parts[1]] = parts[2]

        # Extract road network information
        self.roads = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'road':
                self.roads.add((parts[1], parts[2]))
                self.roads.add((parts[2], parts[1]))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the given state.
        """
        state = node.state
        packages_at_goal = 0
        total_packages = len(self.package_goals)
        heuristic_value = 0

        for package, goal_location in self.package_goals.items():
            package_at_goal = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == package and parts[2] == goal_location:
                    package_at_goal = True
                    break
            if not package_at_goal:
                heuristic_value += 3  # pick-up + drive + drop

        # If all packages are at their goal locations, the heuristic value is 0
        if heuristic_value == 0:
            return 0

        return heuristic_value
