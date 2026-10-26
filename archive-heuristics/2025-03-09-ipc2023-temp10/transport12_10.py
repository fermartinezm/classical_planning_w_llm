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


class transport12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages at the wrong locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic does not consider capacity constraints of vehicles.
    - The heuristic assumes that a vehicle is always available at the package's current location.
    - The heuristic assumes that roads always exist between any two locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Store the road network information to estimate driving costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. Compare the current location of each package with its goal location.
    3. For each package not at its goal location, estimate the number of actions required:
       - 1 pick-up action
       - 1 drop action
       - 1 drive action (to move the vehicle to the goal location)
    4. Sum the estimated number of actions for all packages not at their goal locations.
    5. If the current state is a goal state, return 0.
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
        estimated_cost = 0

        # Check if the current state is a goal state.
        if task.goal_reached(node.state):
            return 0

        # Extract current locations of packages.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at" and args[0] in self.goal_locations:
                package, location = args
                current_locations[package] = location

        # Estimate the number of actions for each package not at its goal.
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations or current_locations[package] != goal_location:
                # Estimate cost: 1 pick-up + 1 drop + 1 drive
                estimated_cost += 3

        return estimated_cost
