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


class transport22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped off at its destination.
    - The heuristic assumes that the vehicles are always available and capable of carrying the packages.
    - The heuristic does not consider capacity constraints.
    - The heuristic assumes that there is a direct road between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current location of each package and vehicle from the state.
    2. For each package, determine its goal location.
    3. If a package is not at its goal location, estimate the number of actions required to move it:
       - One pick-up action to load the package into a vehicle.
       - One drive action to move the vehicle to the goal location.
       - One drop action to unload the package at the goal location.
    4. Sum the estimated number of actions for all packages.
    5. If all packages are at their goal locations, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package = get_parts(goal)[1]
                location = get_parts(goal)[2]
                self.goal_locations[package] = location

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
        package_locations = {}
        vehicle_locations = {}

        # Extract current package locations
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                location = get_parts(fact)[2]
                package_locations[obj] = location
            elif match(fact, "in", "*", "*"):
                package = get_parts(fact)[1]
                vehicle = get_parts(fact)[2]
                # Find the vehicle's location
                for f in state:
                    if match(f, "at", vehicle, "*"):
                        vehicle_location = get_parts(f)[2]
                        package_locations[package] = vehicle_location
                        break  # Vehicle location found, exit inner loop

        # Check if all goals are reached
        all_goals_reached = True
        for package, goal_location in self.goal_locations.items():
            if package not in package_locations or package_locations[package] != goal_location:
                all_goals_reached = False
                break

        if all_goals_reached:
            return 0

        # Estimate the number of actions required
        heuristic_value = 0
        for package, goal_location in self.goal_locations.items():
            if package not in package_locations or package_locations[package] != goal_location:
                heuristic_value += 3  # pick-up + drive + drop

        return heuristic_value
