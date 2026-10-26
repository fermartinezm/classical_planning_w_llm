from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, considering wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip; fnmatch handles the wildcard '*'
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the estimates for all packages. It provides
    a lower bound on the actions needed for each package individually, ignoring
    vehicle capacity and coordination constraints.

    # Assumptions
    - The primary goal is to move packages to specific locations, specified by `(at package location)` goals.
    - Vehicles are the only means of transport for packages between locations.
    - Packages can be on the ground at a location or inside a vehicle.
    - The heuristic assumes a suitable vehicle is available when needed (ignoring capacity and current vehicle location for packages on the ground).
    - The cost of driving between any two connected locations is estimated as 1 action for heuristic purposes.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
      Only `(at package location)` goals are considered.
    - Identify vehicles present in the initial state or goals (e.g., from `capacity` or `in` facts) to distinguish them from packages and locations when parsing state facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each package that has a goal location specified in the task:

    1. Check if the package is already at its goal location in the current state.
       - If `(at package goal_location)` is true in the state, the cost for this package is 0.

    2. If the package is not at its goal location, determine its current status:
       - Build temporary lookup dictionaries for the current state:
         - `package_current_location`: maps package name to its current location name (if on ground).
         - `package_current_vehicle`: maps package name to the vehicle it's in (if in vehicle).
         - `vehicle_current_location`: maps vehicle name to its current location name.
         This is done by iterating through all facts in the current state. Objects are classified as packages or vehicles based on the information gathered during initialization (`self.goal_locations` keys for packages, `self.vehicles` set for vehicles).

       - Using the lookup dictionaries, check if the package is currently inside a vehicle (`package in package_current_vehicle`).
         - If yes, get the vehicle name. Find the current location of that vehicle using `vehicle_current_location[vehicle]`.
           - If the vehicle's current location is the same as the package's `goal_location`, the package needs 1 action (drop).
           - If the vehicle's current location is different from the package's `goal_location`, the package needs at least 2 actions (drive vehicle to goal, then drop).
       - If the package is not inside a vehicle, it must be on the ground (`package in package_current_location`).
         - Since we already checked that it's not at the goal location, it must be at some `current_location` different from the goal.
         - The package needs at least 3 actions: be picked up by a vehicle, be driven to the goal location, and be dropped at the goal location. This assumes a vehicle is available for pickup at the package's current location.

    3. Sum the estimated costs for all packages that are not yet at their goal location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and identifying vehicles.
        """
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        # Assume any object appearing as the first argument of an 'at' goal
        # is a package we need to deliver.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's (at obj loc)
                obj, location = args
                self.goal_locations[obj] = location

        # Identify vehicles. Look for objects in 'capacity' facts or
        # as the second argument of 'in' facts in the initial state or goals.
        self.vehicles = set()
        # Check initial state facts
        for fact in task.initial_state:
             pred, *args = get_parts(fact)
             if pred == 'capacity' and len(args) >= 1:
                 self.vehicles.add(args[0])
             elif pred == 'in' and len(args) >= 2:
                 # args[0] is package, args[1] is vehicle
                 self.vehicles.add(args[1])
        # Check goal facts (less common for vehicles, but possible)
        for goal in self.goals:
             pred, *args = get_parts(goal)
             if pred == 'in' and len(args) >= 2:
                 self.vehicles.add(args[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Build lookup dictionaries for current state for efficient access
        package_current_location = {} # package -> location (if on ground)
        package_current_vehicle = {}  # package -> vehicle (if in vehicle)
        vehicle_current_location = {} # vehicle -> location

        for fact in state:
            pred, *args = get_parts(fact)
            if pred == 'at' and len(args) == 2:
                obj, loc = args
                # Classify object based on pre-identified packages and vehicles
                if obj in self.goal_locations: # It's a package we care about
                     package_current_location[obj] = loc
                elif obj in self.vehicles: # It's a vehicle
                     vehicle_current_location[obj] = loc
                # Other 'at' facts (e.g., at road, at size) are ignored by this heuristic
            elif pred == 'in' and len(args) == 2:
                pkg, veh = args
                # Assume the first arg of 'in' is always a package and second is a vehicle
                package_current_vehicle[pkg] = veh
            # Other predicates like 'capacity', 'road', 'capacity-predecessor' are ignored by this heuristic

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location
            # We check the package_current_location dictionary built from the state
            if package in package_current_location and package_current_location[package] == goal_location:
                # Goal (at package goal_location) is satisfied for this package
                continue

            # If not at goal, determine current status and add cost
            if package in package_current_vehicle:
                # Package is in a vehicle
                vehicle = package_current_vehicle[package]
                # Find the vehicle's location
                if vehicle in vehicle_current_location:
                    current_vehicle_location = vehicle_current_location[vehicle]
                    if current_vehicle_location == goal_location:
                        # Package is in a vehicle, and the vehicle is at the goal location
                        total_cost += 1 # Needs 1 action: drop
                    else:
                        # Package is in a vehicle, but the vehicle is not at the goal location
                        total_cost += 2 # Needs 2 actions: drive vehicle to goal, then drop
                else:
                    # Should not happen in valid states: package in vehicle, but vehicle location unknown
                    # Assign a cost assuming it needs transport and drop
                     total_cost += 2 # Estimate: drive + drop
            elif package in package_current_location:
                # Package is on the ground, not at the goal location (checked at the start of the loop)
                # Needs 3 actions: pick-up, drive, drop
                total_cost += 3
            # If package is neither in a vehicle nor at a location in the state,
            # it's an unexpected state. The current logic implicitly assigns 0 cost
            # if the package isn't found in package_current_location or package_current_vehicle.
            # This is okay for a non-admissible heuristic; it just means this package
            # doesn't contribute to the heuristic value in this unexpected state.
            # In valid states, a package is always either 'at' a location or 'in' a vehicle.


        # The heuristic value is the sum of costs for all packages not at their goal.
        # It is 0 if and only if all packages listed in self.goal_locations are
        # currently at their respective goal locations. This aligns with the goal
        # structure of the transport domain examples.

        return total_cost
