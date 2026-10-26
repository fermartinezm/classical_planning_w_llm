from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Handle cases where pattern is shorter than fact parts, but matches prefix
    # Example: match("(at p1 l1)", "at", "p1") should be False
    # The zip ensures we only compare up to the length of the shorter sequence.
    # We need to ensure all args were consumed.
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the costs for all packages independently.
    It considers the package's current location (on the ground or in a vehicle)
    and the vehicle's location, and estimates the minimum number of pick-up,
    drop, and drive actions needed for that specific package.

    # Assumptions
    - Each pick-up, drop, and drive action costs 1.
    - Capacity constraints are ignored. Any vehicle can pick up any package.
    - Multiple packages can be transported by the same vehicle, but the heuristic
      counts the drive cost for each package independently if they are in the
      same vehicle and need to go to the same location. This might overestimate
      but is simple and acceptable for a greedy best-first search heuristic.
    - The heuristic focuses only on achieving package goal locations.
    - A drive action between any two *different* locations is assumed to cost 1,
      provided a road exists (which is implicitly assumed if transport is possible).
    - If a package is on the ground at a location and no vehicle is currently
      at that location, an extra drive action is added to represent a vehicle
      moving to the package's location for pickup.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Identifies all vehicles present in the initial state to track their locations.

    # Step-by-Step Thinking for Computing Heuristic
    For each package that is not yet at its goal location:

    1. Determine the package's current status: Is it on the ground at a location `l_c`,
       or is it inside a vehicle `v`?
    2. If the package is on the ground at `l_c`:
       - If `l_c` is the goal location, the cost for this package is 0.
       - If `l_c` is not the goal location:
         - It needs to be picked up (1 action).
         - It needs to be transported by a vehicle to the goal location (at least 1 drive action).
         - It needs to be dropped at the goal location (1 action).
         - Total minimum actions = 3 (pick-up + drive + drop).
         - Additionally, check if *any* vehicle is currently at `l_c`. If no vehicle is present,
           an extra drive action is needed for a vehicle to reach `l_c` to pick up the package.
           Add 1 to the cost if no vehicle is at `l_c`.
         - Total cost for this package = 3 or 4.
    3. If the package is inside a vehicle `v`:
       - Find the current location `l_v` of vehicle `v`.
       - If `l_v` is the goal location:
         - The package only needs to be dropped (1 action).
         - Total cost for this package = 1.
       - If `l_v` is not the goal location:
         - The vehicle needs to drive to the goal location (at least 1 drive action).
         - The package needs to be dropped at the goal location (1 action).
         - Total minimum actions = 2 (drive + drop).
         - Total cost for this package = 2.
    4. The total heuristic value is the sum of the costs calculated for each
       misplaced package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and identifying vehicles.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages that are goals
        goal_packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                goal_packages.add(package)

        # Identify all vehicles from the initial state.
        # Vehicles appear as the first argument of 'at' or 'capacity',
        # or the second argument of 'in'.
        self.all_vehicles = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # If it's not a package we care about (i.e., in goals), assume it's a vehicle
                 # This is an inference based on common domain structure.
                 if obj not in goal_packages:
                     self.all_vehicles.add(obj)
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                self.all_vehicles.add(vehicle)
            elif parts[0] == 'capacity':
                 vehicle, size = parts[1], parts[2]
                 self.all_vehicles.add(vehicle)

        # Ensure any vehicles mentioned in goals (less common) are also included
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj not in goal_packages: # If it's not a package goal, assume it's a vehicle goal
                     self.all_vehicles.add(obj)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # package_status: maps package -> ('at', loc) or ('in', vehicle)
        package_status = {}
        # vehicle_locations: maps vehicle -> loc
        vehicle_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we need to deliver
                    package_status[obj] = ('at', loc)
                elif obj in self.all_vehicles: # It's a vehicle
                    vehicle_locations[obj] = loc
                # Ignore 'at' facts for objects that are not goal packages or known vehicles
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # It's a package we need to deliver
                    package_status[package] = ('in', vehicle)
                # Ignore 'in' facts for packages not in goals

        total_cost = 0  # Initialize action cost counter.

        # Check if any vehicle is currently at any location
        locations_with_vehicles = set(vehicle_locations.values())

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state at all, something is wrong,
            # or it was perhaps consumed? Assuming valid states where packages exist.
            if package not in package_status:
                 # This case indicates an unexpected state structure.
                 # Return infinity to prune this path, or handle based on domain specifics.
                 # For transport, packages aren't consumed, so this shouldn't happen.
                 # Let's assume valid states and proceed.
                 # print(f"Warning: Package {package} not found in state.")
                 continue # Skip this package, or handle as error

            status, loc_or_veh = package_status[package]

            # Case 1: Package is on the ground
            if status == 'at':
                current_location = loc_or_veh
                if current_location != goal_location:
                    # Package needs pickup, drive, drop
                    cost_for_package = 3 # pick + drive + drop

                    # Check if a vehicle is available at the current location for pickup
                    if current_location not in locations_with_vehicles:
                        cost_for_package += 1 # Need to drive a vehicle to the pickup location

                    total_cost += cost_for_package

            # Case 2: Package is inside a vehicle
            elif status == 'in':
                vehicle = loc_or_veh
                vehicle_location = vehicle_locations.get(vehicle)

                if vehicle_location is None:
                    # Vehicle containing the package is not at any location.
                    # This indicates an invalid state representation according to the domain.
                    # Return infinity to prune this path.
                    # print(f"Error: Vehicle {vehicle} containing {package} is not at any location.")
                    return float('inf')

                if vehicle_location != goal_location:
                    # Vehicle needs to drive to goal, then package needs drop
                    total_cost += 2 # drive + drop
                else: # vehicle_location == goal_location
                    # Package needs drop
                    total_cost += 1 # drop

        return total_cost

# Helper functions get_parts and match are defined above the class.
