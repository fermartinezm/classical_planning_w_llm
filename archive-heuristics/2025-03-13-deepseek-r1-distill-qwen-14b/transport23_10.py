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
    - Vehicles can move along roads between locations.
    - Each vehicle has a specific capacity that determines how many packages it can carry.

    # Heuristic Initialization
    - Extract goal locations for each package.
    - Extract static facts to determine vehicle capacities and road connections.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each package, check if it is already at its goal location. If it is, no actions are needed.
    2. If the package is not at its goal location, determine if it is inside a vehicle or on the ground.
    3. If the package is inside a vehicle, check if the vehicle is at the correct location to drop the package. If not, calculate the number of drive actions needed to move the vehicle to the correct location.
    4. If the package is on the ground, calculate the number of actions needed to pick up the package, drive to the goal location, and drop the package.
    5. Sum the actions for all packages, considering the vehicle's capacity to carry multiple packages per trip.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Vehicle capacities from static facts.
        - Road connections between locations.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

        # Extract vehicle capacities
        self.vehicle_capacities = {}
        for fact in static_facts:
            if match(fact, "capacity", "*", "*"):
                _, vehicle, capacity = get_parts(fact)
                self.vehicle_capacities[vehicle] = capacity

        # Extract road connections
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                self.roads.add(fact)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track current locations of packages and vehicles
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_actions = 0  # Initialize action counter

        # For each package, check if it's at the goal location
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations:
                # Package is not present in the state (should not happen in valid state)
                continue
            current_location = current_locations[package]

            if current_location == goal_location:
                continue  # Package is already at the goal

            # Determine if the package is inside a vehicle
            in_vehicle = current_location not in self.vehicle_capacities
            if in_vehicle:
                # The package is inside a vehicle; find the vehicle's location
                vehicle = None
                for fact in state:
                    if match(fact, "in", package, "*"):
                        vehicle = get_parts(fact)[1]
                        break
                if vehicle:
                    # Vehicle's current location
                    vehicle_location = current_locations[vehicle]
                    if vehicle_location != goal_location:
                        # Need to drive the vehicle to the goal location
                        total_actions += 1  # Drive action
                    # Drop the package at the goal location
                    total_actions += 1
                else:
                    # This case should not occur in a valid state
                    pass
            else:
                # The package is on the ground; need to pick it up, drive, and drop it
                total_actions += 1  # Pick-up action
                total_actions += 1  # Drive action
                total_actions += 1  # Drop action

        return total_actions
