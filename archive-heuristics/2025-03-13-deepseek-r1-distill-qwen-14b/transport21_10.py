from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each package must be transported independently, assuming no shared rides.
    - The heuristic assumes optimal vehicle routing between locations.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Stores static facts including road connections and vehicle capacities.

    # Step-by-Step Thinking for Computing Heuristic Value
    1. Extract goal locations for each package from the problem's goal conditions.
    2. For each package, determine its current location and whether it is inside a vehicle.
    3. If the package is not at its goal location:
       - If the package is not in a vehicle, estimate 2 actions (pickup + drive).
       - If the package is already in a vehicle, estimate 1 action (drive).
    4. Sum the estimated actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections and vehicle capacities).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

        # Store road connections for quick lookup
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1))

        # Store vehicle capacities
        self.vehicle_capacities = {}
        for fact in static_facts:
            if match(fact, "capacity", "*", "*"):
                vehicle, size = get_parts(fact)[1], get_parts(fact)[2]
                self.vehicle_capacities[vehicle] = size

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track where packages and vehicles are currently located
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_actions = 0  # Initialize action counter

        # Process each package
        for package in self.goal_locations:
            goal_location = self.goal_locations[package]

            # Skip if already at goal
            if package in current_locations and current_locations[package] == goal_location:
                continue

            # Determine if the package is in a vehicle
            in_vehicle = False
            vehicle = None
            for fact in state:
                if match(fact, "in", package, "*"):
                    vehicle = get_parts(fact)[1]
                    in_vehicle = True
                    break

            if in_vehicle:
                # Package is already in a vehicle; just need to drive to goal
                total_actions += 1  # Drive action
            else:
                # Package is on the ground; need to pick up and drive
                total_actions += 2  # Pick-up + drive actions

        return total_actions
