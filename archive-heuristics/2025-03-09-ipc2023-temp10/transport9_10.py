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


class transport9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and the number of vehicles available.
    It also takes into account the need to drive vehicles to locations where packages need to be picked up or dropped off.

    # Assumptions
    - Each package needs to be picked up and dropped off at its destination.
    - Vehicles need to drive to the locations where packages are.
    - The heuristic does not consider capacity constraints.

    # Heuristic Initialization
    - Extract the goal locations for each package.
    - Identify the locations of all vehicles.
    - Store the road network information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages and vehicles from the state.
    2. For each package, check if it is at its goal location.
    3. Count the number of packages that are not at their goal locations.
    4. Estimate the number of pick-up and drop actions needed (at least one pick-up and one drop for each misplaced package).
    5. Estimate the number of drive actions needed to move vehicles to packages and goal locations.
    6. Return the sum of these estimates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Locations of all vehicles.
        - Road network information.
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
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current locations of packages and vehicles
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                location = get_parts(fact)[2]
                if obj in self.goal_locations:  # It's a package
                    package_locations[obj] = location
                else:  # It's a vehicle
                    vehicle_locations[obj] = location
            elif match(fact, "in", "*", "*"):
                package = get_parts(fact)[1]
                vehicle = get_parts(fact)[2]
                vehicle_locations[package] = vehicle

        # Count misplaced packages
        misplaced_packages = 0
        for package, location in package_locations.items():
            if package not in self.goal_locations or location != self.goal_locations[package]:
                misplaced_packages += 1

        # Estimate the number of pick-up and drop actions
        pick_drop_actions = misplaced_packages * 2

        # Estimate the number of drive actions
        drive_actions = 0
        # For each misplaced package, estimate the number of drive actions needed to get a vehicle to the package and to the goal location
        # This is a very rough estimate and can be improved by considering the actual road network
        drive_actions = misplaced_packages #* 2  # Assume at least one drive action per misplaced package

        # Return the sum of these estimates
        return pick_drop_actions + drive_actions
