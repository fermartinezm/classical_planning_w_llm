from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct based on examples

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully, although PDDL states are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated minimum actions
    required for each package to reach its individual goal location, ignoring vehicle capacity
    and specific road network distances.

    # Assumptions
    - Each package's movement towards its goal is considered independently.
    - Vehicle capacity constraints are ignored.
    - Reachability between locations via roads is assumed whenever a drive action is needed.
    - The cost of a 'drive' action is considered a single unit for heuristic purposes, regardless of distance.
    - A package is either at a location or inside a vehicle, but not both.
    - States are valid, meaning packages are either at a location or in a vehicle, and vehicles are always at a location.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a specified goal location:
    1. Check if the package is currently at its goal location. If yes, the cost for this package is 0.
    2. If the package is currently at a location but it is NOT the goal location:
       It needs to be picked up, transported by a vehicle, and dropped at the goal.
       This requires at least 3 actions (pick-up, drive, drop). Cost for this package is 3.
    3. If the package is currently inside a vehicle:
       a. Find the current location of the vehicle carrying the package.
       b. If the vehicle is at the package's goal location:
          The package only needs to be dropped. Cost for this package is 1.
       c. If the vehicle is at a location that is NOT the package's goal location:
          The vehicle needs to drive to the goal location, and then the package needs to be dropped.
          This requires at least 2 actions (drive, drop). Cost for this package is 2.
    4. The total heuristic value is the sum of the costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages.
        """
        self.goals = task.goals
        self.goal_locations = {}
        # Extract goal locations for packages from the goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at package location)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to get all packages to their goal locations.
        """
        state = node.state
        total_cost = 0

        # Map current state of locatable objects (packages and vehicles)
        current_locations = {} # Maps object -> location (if at a location)
        in_vehicle_status = {} # Maps package -> vehicle (if in a vehicle)

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts if any
                continue
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                in_vehicle_status[package] = vehicle

        # Calculate cost for each package that has a goal
        for package, goal_location in self.goal_locations.items():
            cost_for_package = 0 # Default cost if already satisfied

            # Check if package is currently at its goal location
            if package in current_locations and current_locations[package] == goal_location:
                cost_for_package = 0 # Package is already at its goal

            # Check if package is at a non-goal location
            elif package in current_locations and current_locations[package] != goal_location:
                # Package is on the ground at the wrong location.
                # Needs: Pick-up (1), Drive (>=1, estimate 1), Drop (1) = 3 actions minimum.
                cost_for_package = 3

            # Check if package is inside a vehicle
            elif package in in_vehicle_status:
                vehicle = in_vehicle_status[package]
                # Find the vehicle's current location
                if vehicle in current_locations:
                    vehicle_location = current_locations[vehicle]
                    if vehicle_location == goal_location:
                        # Package is in a vehicle which is at the goal location.
                        # Needs: Drop (1) = 1 action minimum.
                        cost_for_package = 1
                    else: # vehicle_location != goal_location
                        # Package is in a vehicle which is at the wrong location.
                        # Needs: Drive (>=1, estimate 1), Drop (1) = 2 actions minimum.
                        cost_for_package = 2
                # else: # Vehicle is not at any location - this should not happen in valid states
                    # cost_for_package = 2 # Estimate drive + drop

            # else: # Package is neither 'at' nor 'in' - this should not happen in valid states
                # cost_for_package = 3 # Estimate pick-up + drive + drop

            total_cost += cost_for_package

        return total_cost
