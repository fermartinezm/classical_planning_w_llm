<code-file-heuristic-transport>
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
    This heuristic estimates the number of actions needed to transport all packages to their goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Each vehicle has a specific capacity that determines how many packages it can carry.
    - Packages must be picked up from their current location and dropped off at their goal location.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goals.
    - Extract static facts including road connections and vehicle capacities.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each package, determine its current location and whether it is already at the goal.
    2. For packages not at their goal:
       a. Check if they are inside a vehicle or on the ground.
       b. If inside a vehicle, determine the vehicle's current location.
       c. If on the ground, check if a vehicle is available at the same location.
    3. Calculate the required actions to move the package to its goal:
       a. If the package is in a vehicle, calculate the drive actions needed to reach the goal location.
       b. If the package is on the ground, calculate the actions needed to pick it up and then drop it off at the goal.
    4. Sum the actions for all packages, considering vehicle capacities to minimize the number of trips.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections and vehicle capacities).
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

        # Extract vehicle capacities from static facts
        self.vehicle_capacities = {}
        for fact in static_facts:
            if match(fact, "capacity", "*", "*"):
                vehicle, size = get_parts(fact)[1], get_parts(fact)[2]
                self.vehicle_capacities[vehicle] = size

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

        # Track locations of packages and vehicles
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter

        # Track which packages have been accounted for
        processed_packages = set()

        # Find the vehicle that will be used for transportation
        # Assume we have only one vehicle for simplicity
        vehicle = None
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.vehicle_capacities:
                vehicle = get_parts(fact)[1]
                vehicle_location = get_parts(fact)[2]
                break

        if not vehicle:
            # No vehicle available, heuristic is infinity (problem unsolvable)
            return float('inf')

        # Determine the current location of the vehicle
        vehicle_current = None
        for fact in state:
            if match(fact, "at", vehicle, "*"):
                vehicle_current = get_parts(fact)[2]
                break

        # Process each package
        for package, goal_location in self.goal_locations.items():
            if package in processed_packages:
                continue

            # Check if package is already at goal
            if current_locations.get(package, None) == goal_location:
                continue

            # Determine if package is in a vehicle or on the ground
            package_location = current_locations.get(package, None)
            if package_location and package_location != goal_location:
                # Package is on the ground
                # Need to pick up and drop off
                total_cost += 2  # 1 for pickup, 1 for dropoff

                # Check if vehicle is already at the package's location
                if vehicle_current != package_location:
                    # Vehicle needs to drive to package location
                    # Find path length (assume unit cost per drive action)
                    # For simplicity, assume direct connection
                    total_cost += 1  # drive action

                    # After pickup, vehicle needs to drive to goal location
                    if vehicle_current != goal_location:
                        total_cost += 1  # drive action
                else:
                    # Vehicle is already at package location
                    # Just need to drop off
                    total_cost += 1  # drive action

                processed_packages.add(package)

        return total_cost
</code-file-heuristic-transport>