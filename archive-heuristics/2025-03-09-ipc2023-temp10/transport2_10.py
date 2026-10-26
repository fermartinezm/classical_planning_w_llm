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


class transport2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the number of packages that are not at their goal locations and estimates the number of pick-up, drop, and drive actions needed.

    # Assumptions
    - Each package needs to be picked up, transported, and dropped at its destination.
    - The heuristic does not take into account vehicle capacity or size constraints.
    - The heuristic assumes that there are always roads between locations.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Store the road connections between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current locations of all packages from the state.
    2. Compare the current location of each package with its goal location.
    3. For each package not at its goal location, estimate the number of actions required:
       - 1 pick-up action
       - 1 drive action (to move the vehicle to the goal location)
       - 1 drop action
    4. Sum the estimated number of actions for all packages not at their goal locations.
    5. If all packages are at their goal locations, return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each package and road connections.
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

        # Extract road connections between locations
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                location1 = get_parts(fact)[1]
                location2 = get_parts(fact)[2]
                self.roads.add((location1, location2))

    def __call__(self, node):
        """
        Compute an estimate of the number of actions required to reach the goal state from the given state.
        """
        state = node.state
        packages_at_goal = 0
        total_packages = len(self.goal_locations)
        estimated_actions = 0

        for package, goal_location in self.goal_locations.items():
            current_location = None
            # Check if the package is at a location
            for fact in state:
                if match(fact, "at", package, "*"):
                    current_location = get_parts(fact)[2]
                    break
                elif match(fact, "in", package, "*"):
                    # If the package is in a vehicle, find the vehicle's location
                    vehicle = get_parts(fact)[2]
                    for vehicle_fact in state:
                        if match(vehicle_fact, "at", vehicle, "*"):
                            current_location = get_parts(vehicle_fact)[2]
                            break
                    break

            if current_location == goal_location:
                packages_at_goal += 1
            else:
                estimated_actions += 3  # pick-up, drive, drop

        if packages_at_goal == total_packages:
            return 0  # All packages are at their goal locations
        else:
            return estimated_actions
