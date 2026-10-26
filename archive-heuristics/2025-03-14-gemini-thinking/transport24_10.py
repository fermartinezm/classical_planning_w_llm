from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(road l1 l2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to transport all packages to their goal locations.
    It considers the necessary pick-up, drop, and drive actions. The heuristic is based on the number of road connections
    needed to reach the goal location for each package, plus actions for loading and unloading.

    # Assumptions:
    - Each package needs to be picked up, transported, and dropped at its goal location.
    - The heuristic does not explicitly consider vehicle capacity or optimal vehicle assignment.
    - It assumes that a path exists between the current location and the goal location if they are different.
    - The cost is based on the number of actions, assuming each action (drive, pick-up, drop) has a unit cost.

    # Heuristic Initialization
    - Extracts the goal predicates to determine the target location for each package.
    - Extracts static `road` predicates to understand the connectivity of locations.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that is not at its goal location:
    1. Determine the current location of the package.
    2. Determine the goal location of the package from the goal predicates.
    3. If the package is not at its goal location:
        a. If the package is not currently in a vehicle, estimate the cost of a 'pick-up' action (cost: 1).
        b. Estimate the number of 'drive' actions needed to move from the current location to the goal location.
           This is simplified to 1 if the current and goal locations are different and there is a road connection (or path).
           A more sophisticated approach could involve shortest path algorithms, but for efficiency, we simplify it.
           For this simplified heuristic, we assume a direct road exists if locations are different and reachable.
           Thus, if current location and goal location are different, we add 1 for 'drive'.
        c. Estimate the cost of a 'drop' action at the goal location (cost: 1).
    4. Sum up the estimated costs for all packages.
    5. If all packages are at their goal locations, the heuristic value is 0.

    This heuristic is a simplification and does not guarantee admissibility, but it aims to provide a reasonable estimate
    of the remaining actions to guide the search efficiently in the Transport domain.
    """

    def __init__(self, task):
        """
        Initialize the transport heuristic.

        - Extracts goal locations for each package.
        - Extracts road network information from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "?p - package", "?l - location"):
                parts = get_parts(goal)
                package = parts[1]
                location = parts[2]
                self.package_goals[package] = location

        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "?l1 - location", "?l2 - location"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1)) # Roads are bidirectional in the examples

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        For each package not at its goal location, estimate the number of actions:
        - pick-up (if not in vehicle)
        - drive (if current location != goal location)
        - drop
        Sum these costs for all packages.
        """
        state = node.state
        heuristic_value = 0

        for package, goal_location in self.package_goals.items():
            current_location = None
            in_vehicle = False
            vehicle_location = None

            for fact in state:
                if match(fact, "at", package, "*"):
                    current_location = get_parts(fact)[2]
                    break
                elif match(fact, "in", package, "*"):
                    in_vehicle = True
                    vehicle = get_parts(fact)[2]
                    for vehicle_fact in state:
                        if match(vehicle_fact, "at", vehicle, "*"):
                            vehicle_location = get_parts(vehicle_fact)[2]
                            current_location = vehicle_location # Package is at vehicle location
                            break
                    break

            if current_location != goal_location:
                if not in_vehicle:
                    heuristic_value += 1  # pick-up cost

                if current_location is not None and goal_location is not None and current_location != goal_location:
                    heuristic_value += 1  # drive cost (simplified to 1 if locations are different)

                heuristic_value += 1  # drop cost

        return heuristic_value
