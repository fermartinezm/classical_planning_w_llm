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


class transport1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of pick-up,
    drop, and drive actions needed to transport them.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic does not take into account vehicle capacity or size constraints.
    - The heuristic assumes that a vehicle is always available at the package's current location.
    - The heuristic assumes that there is a direct road between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each package from the state.
    2. Compare the current location of each package with its goal location.
    3. For each package not at its goal location, estimate the number of actions required:
        - 1 pick-up action
        - 1 drop action
        - 1 drive action (to move the vehicle to the goal location)
    4. Sum the estimated number of actions for all packages not at their goal locations.
    5. If all packages are at their goal locations, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package = get_parts(goal)[1]
                location = get_parts(goal)[2]
                self.package_goals[package] = location

        # Extract road network information
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1 = get_parts(fact)[1]
                l2 = get_parts(fact)[2]
                self.roads.add((l1, l2))

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the given state.
        """
        state = node.state
        packages_at_goal = 0
        total_cost = 0

        # Extract current package locations
        package_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                package = get_parts(fact)[1]
                location = get_parts(fact)[2]
                package_locations[package] = location

        # Compare current locations with goal locations
        for package, goal_location in self.package_goals.items():
            if package in package_locations:
                current_location = package_locations[package]
                if current_location != goal_location:
                    # Estimate cost for pick-up, drive, and drop
                    total_cost += 3  # 1 pick-up + 1 drive + 1 drop
                else:
                    packages_at_goal += 1
            else:
                # Package is in a vehicle, need to unload, drive and drop
                total_cost += 3

        # If all packages are at their goal locations, return 0
        if packages_at_goal == len(self.package_goals):
            return 0

        return total_cost
