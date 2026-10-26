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


class transport24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of pick-up,
    drop, and drive actions needed to transport them. It also takes into account the current location of vehicles
    and packages.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - Vehicles can carry only one package at a time (simplified assumption).
    - The heuristic does not consider capacity constraints.
    - The heuristic assumes that there is always a path between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Extract the road network from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract Goal Information:
       - Identify the goal location for each package.

    2. Identify Current State:
       - Determine the current location of each package and vehicle.

    3. Calculate Heuristic Value:
       - For each package not at its goal location:
         - Estimate the number of actions needed to move the package to its goal.
           This includes:
             - Finding a vehicle at the package's current location.
             - Picking up the package.
             - Driving the vehicle to the package's goal location.
             - Dropping the package at the goal location.
       - Sum the estimated number of actions for all packages.
       - If the current state is a goal state, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package = get_parts(goal)[1]
                location = get_parts(goal)[2]
                self.package_goals[package] = location

        # Extract road network information.
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

        # Check if the current state is a goal state.
        if self.goal_reached(state):
            return 0

        # Extract current locations of packages and vehicles.
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                location = get_parts(fact)[2]
                if obj.startswith("p"):
                    package_locations[obj] = location
                elif obj.startswith("v"):
                    vehicle_locations[obj] = location

        # Calculate the heuristic value.
        heuristic_value = 0
        for package, goal_location in self.package_goals.items():
            if package not in package_locations or package_locations[package] != goal_location:
                # Package is not at its goal location.
                heuristic_value += 4  # pick-up + drive + drop

        return heuristic_value

    def goal_reached(self, state):
        """Check if the current state satisfies all goal conditions."""
        return self.goals <= state
