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
    - Each package requires exactly one action to be picked up and one action to be dropped off.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Parses static facts to identify road connections and capacity relationships.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each package, check if it is already at its goal location. If it is, no actions are needed.
    2. If the package is inside a vehicle, determine the vehicle's current location.
    3. If the package is on the ground, determine its current location.
    4. Calculate the number of actions required to move the package to its goal location:
       - If the package is inside a vehicle, check if the vehicle is already at the package's location. If not, add actions to move the vehicle to the package's location, pick up the package, move to the goal location, and drop the package.
       - If the package is on the ground, add actions to move a vehicle to the package's location, pick up the package, move to the goal location, and drop the package.
    5. Sum the actions required for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and static facts.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract road connections from static facts
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1))

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

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

        # For each package, determine if it's at the goal location
        for package, goal_location in self.goal_locations.items():
            # Skip if the package is already at the goal
            if f"(at {package} {goal_location})" in state:
                continue

            # Determine if the package is inside a vehicle or on the ground
            in_vehicle = False
            vehicle = None
            for fact in state:
                if match(fact, "in", package, "*"):
                    in_vehicle = True
                    vehicle = get_parts(fact)[1]
                    break

            if in_vehicle:
                # Package is inside a vehicle; find the vehicle's location
                vehicle_location = current_locations.get(vehicle, None)
                if vehicle_location is None:
                    continue  # Vehicle not found, skip for now
                # Check if the vehicle is already at the goal location
                if vehicle_location == goal_location:
                    continue  # No action needed
                else:
                    # Actions needed: drive to package's location, pick up, drive to goal, drop
                    total_actions += 4  # drive, pick-up, drive, drop
            else:
                # Package is on the ground; find its location
                package_location = current_locations.get(package, None)
                if package_location is None:
                    continue  # Package not found, skip for now
                if package_location == goal_location:
                    continue  # No action needed
                else:
                    # Actions needed: drive to package's location, pick up, drive to goal, drop
                    total_actions += 4  # drive, pick-up, drive, drop

        return total_actions
