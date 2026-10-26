from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like '(predicate arg1 arg2 ...)'
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move each package
    to its goal location. It sums the estimated costs for each package
    independently. The estimate for a single package is based on its current
    status (on the ground or in a vehicle) and its effective location relative
    to its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location specified in the task goals.
    - The cost for moving a package from its current effective location
      to its goal location via a vehicle is estimated as a fixed number
      of actions (pick-up, drive, drop), ignoring actual path distance,
      vehicle availability, and capacity constraints.
    - A 'drive trip' between any two distinct locations is estimated as 1 action
      for heuristic calculation purposes.
    - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles.

    # Heuristic Initialization
    - The constructor extracts the goal locations for each package from the task goals.
      Static facts like road network or capacity predecessors are not used in this
      simplified heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Check if the state is a goal state. If all goal conditions (specifically,
       all required `(at package goal_location)` facts) are met, the heuristic is 0.
    2. If it's not a goal state, initialize the total heuristic cost to 0.
    3. Identify the current status and location of all packages and vehicles in the state
       by parsing the `(at ?x ?l)` and `(in ?p ?v)` facts. Store package status
       (on ground at location, or in vehicle) and vehicle locations.
    4. Iterate through each package that has a goal location defined in the task.
    5. For the current package, check if the goal fact `(at package goal_location)`
       is already present in the state. If yes, this package contributes 0 to the
       total heuristic cost.
    6. If the goal fact is not present, determine the package's effective current location:
       - If the package is on the ground at `l_current` (`(at package l_current)` in state),
         its effective location is `l_current`.
       - If the package is inside a vehicle `v` (`(in package v)` in state), find the
         vehicle's location `l_v` (`(at v l_v)` in state). The package's effective
         location is `l_v`.
    7. Based on the package's status and effective location relative to its goal location `l_goal`,
       add an estimated cost for this package to the total heuristic:
       - If the package is on the ground at `l_current` (`l_current != l_goal`):
         It needs to be picked up (1 action), the vehicle needs to drive from `l_current`
         to `l_goal` (estimated as 1 action), and it needs to be dropped (1 action).
         Estimated cost for this package: 1 + 1 + 1 = 3.
       - If the package is inside a vehicle `v` which is at `l_v`:
         - If `l_v == l_goal`: The package needs to be dropped (1 action).
           Estimated cost for this package: 1.
         - If `l_v != l_goal`: The vehicle needs to drive from `l_v` to `l_goal`
           (estimated as 1 action), and the package needs to be dropped (1 action).
           Estimated cost for this package: 1 + 1 = 2.
    8. After iterating through all packages with goals, the accumulated total cost
       is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each package.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Assuming goals are always (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                # Assuming package names start with 'p' based on convention
                if package.startswith('p'):
                    self.package_goals[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Quickly check if goal is reached.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located or contained.
        # package_status: {package: ('at', loc) or ('in', vehicle)}
        # vehicle_locations: {vehicle: loc}
        package_status = {}
        vehicle_locations = {}

        # Populate status based on the state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Use naming convention to distinguish packages and vehicles
                if obj.startswith('p'):
                     package_status[obj] = ('at', loc)
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc
                # Ignore other 'at' facts (e.g., static locatable objects)
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                # Use naming convention
                if package.startswith('p') and vehicle.startswith('v'):
                     package_status[package] = ('in', vehicle)
                # Ignore other 'in' facts

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the goal fact is already satisfied
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at its goal location on the ground

            # If not at goal, determine current status and effective location
            current_status = package_status.get(package)

            # This case implies the package exists in goals but not in the state's
            # location/containment facts. This shouldn't happen in valid states.
            # We'll skip it assuming valid state representation.
            if current_status is None:
                 continue

            status_type, obj_or_loc = current_status

            if status_type == 'at':
                # Package is on the ground at obj_or_loc (which is l_current)
                # l_current must be different from goal_location because we already checked the goal fact
                # Needs pick-up (1), drive (1), drop (1)
                total_cost += 3
            elif status_type == 'in':
                # Package is in vehicle obj_or_loc (which is vehicle)
                vehicle = obj_or_loc
                l_v = vehicle_locations.get(vehicle) # Get vehicle's location

                # If vehicle location is not found, state is likely invalid.
                # Assume valid states where vehicles are always at a location.
                if l_v is None:
                    continue

                # Package is in vehicle at l_v
                if l_v == goal_location:
                    # Vehicle is at the goal location, just needs drop
                    total_cost += 1
                else:
                    # Vehicle needs to drive to goal, then drop
                    total_cost += 1 # drive (estimated)
                    total_cost += 1 # drop
            # else: invalid status_type - should not happen with robust parsing

        return total_cost
