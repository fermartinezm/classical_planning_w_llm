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


class transport5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of pick-up,
    drop, and drive actions needed to transport them.

    # Assumptions
    - Each package requires at least one pick-up and one drop action.
    - Vehicles are always available to transport packages.
    - The heuristic does not consider capacity constraints or size dependencies.
    - The heuristic assumes that a vehicle needs to drive to the package and then to the goal.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network information from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Information:
       - Identify the goal location for each package.

    2. Identify Packages Not At Goal:
       - Determine which packages are not currently at their goal locations.

    3. Estimate Actions for Each Package:
       - For each package not at its goal:
         - Assume one pick-up action is needed.
         - Assume one drop action is needed.
         - Assume the vehicle needs to drive to the package location.
         - Assume the vehicle needs to drive to the goal location.

    4. Sum the Estimated Actions:
       - Sum the estimated actions for all packages to get the total heuristic value.
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

        # Extract road information.
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
        packages_at_goal = 0
        total_actions = 0

        for package, goal_location in self.goal_locations.items():
            package_location = None
            for fact in state:
                if match(fact, "at", package, "*"):
                    package_location = get_parts(fact)[2]
                    break
            if package_location is None:
                # Package is in a vehicle, find the vehicle location
                for fact in state:
                    if match(fact, "in", package, "*"):
                        vehicle = get_parts(fact)[2]
                        for fact2 in state:
                            if match(fact2, "at", vehicle, "*"):
                                vehicle_location = get_parts(fact2)[2]
                                package_location = vehicle_location
                                break
                        break
            if package_location == goal_location:
                packages_at_goal += 1
            else:
                # Estimate actions: pick-up, drop, and drives
                total_actions += 3  # pick-up, drop, drive

        # If all packages are at their goal locations, return 0.
        if packages_at_goal == len(self.goal_locations):
            return 0

        return total_actions
