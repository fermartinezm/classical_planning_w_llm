from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings.
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the input is a non-empty string starting and ending with parentheses
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return an empty list for malformed or invalid fact strings
        return []
    # Remove the outer parentheses and split the string by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The fact matches the pattern only if they have the same number of components
    # and each component matches the corresponding pattern argument.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location. It sums the estimated costs for each package
    independently, based on its current state relative to its goal. The
    estimation for a single package is based on the "stages" of transport
    needed: being on the ground at the wrong location, being in a vehicle
    at the wrong location, or being in a vehicle at the correct location.

    # Assumptions
    - Each package's transport can be considered independently of other packages.
    - Vehicle capacity and availability are not explicitly modeled in the cost
      calculation; it assumes a suitable vehicle is generally available when needed
      for pick-up or transport.
    - Driving between any two different locations is estimated as a single
      'drive' action, regardless of the actual road network distance or
      intermediate locations required.
    - All actions (pick-up, drop, drive) are assumed to have a cost of 1.
    - The goal state consists solely of packages being at specific locations.

    # Heuristic Initialization
    - The constructor extracts the goal location for each package from the task's
      goal conditions. It specifically looks for `(at package location)` predicates
      in the goal set. Only packages mentioned in these goal predicates contribute
      to the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that has a specified goal location `goal_loc` in the task's goals:

    1. Determine the current state of package `p` by examining the facts in the current state:
       - Is `p` on the ground at some location `current_loc`? This is indicated by a fact `(at p current_loc)`.
       - Is `p` inside a vehicle `v`? This is indicated by a fact `(in p v)`.

    2. Based on the package's current state and its goal location `goal_loc`, estimate the minimum number of actions required for this package:
       - If the state contains `(at p goal_loc)`: The package is already at its goal location. The estimated cost for this package is 0.
       - If the state contains `(at p current_loc)` where `current_loc != goal_loc`: The package is on the ground but at the wrong location. It needs to be picked up, transported by a vehicle, and dropped at the goal. The estimated cost is 3 actions (1 for pick-up + 1 for drive + 1 for drop).
       - If the state contains `(in p v)` for some vehicle `v`: The package is inside a vehicle. We need to find the vehicle's current location `vehicle_loc` by looking for `(at v vehicle_loc)` in the state.
         - If the vehicle's location `vehicle_loc` is the same as the package's goal location `goal_loc`: The package is in a vehicle that is already at the destination. It just needs to be dropped. The estimated cost is 1 action (1 for drop).
         - If the vehicle's location `vehicle_loc` is different from the package's goal location `goal_loc`: The package is in a vehicle, but the vehicle is at the wrong location. The vehicle needs to drive to the goal location, and then the package needs to be dropped. The estimated cost is 2 actions (1 for drive + 1 for drop).
       - If the package's location is not found in the state (neither `(at p ...)` nor `(in p ...)`), which indicates an unexpected or invalid state representation, a default high cost (e.g., 3) is assigned as a penalty or estimate for a full transport cycle.

    3. The total heuristic value for the state is the sum of the estimated costs calculated for each package that has a goal location and is not yet at that goal location.

    The heuristic value is 0 if and only if all packages specified in the goal are currently located at their respective goal locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages from the task.
        """
        # Store the goal conditions from the task.
        self.goals = task.goals  # frozenset of strings

        # Store goal locations for each package in a dictionary {package_name: goal_location_name}.
        # We only care about packages that need to be at a specific location according to the goal.
        self.package_goals = {}
        for goal in self.goals:
            # Parse the goal fact string. Expected format is typically '(at package location)'.
            parts = get_parts(goal)
            # Check if the parsed fact is an 'at' predicate with exactly two arguments (package and location).
            if len(parts) == 3 and parts[0] == "at":
                package, location = parts[1], parts[2]
                # Store the goal location for this package.
                self.package_goals[package] = location
            # Other types of goal predicates (if any) are ignored by this heuristic.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal state.

        Args:
            node: The current search node, containing the state (frozenset of strings).

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state  # The current world state as a frozenset of strings.

        # Create mappings to quickly find the current location of packages and vehicles.
        # Packages can be on the ground ('at') or inside a vehicle ('in').
        # Vehicles are always on the ground ('at').
        current_locations = {}  # Maps object_name -> location_name for objects on the ground.
        package_in_vehicle = {} # Maps package_name -> vehicle_name for packages inside vehicles.

        # Iterate through the facts in the current state to populate the location mappings.
        for fact in state:
            parts = get_parts(fact)
            # Skip any fact strings that couldn't be parsed.
            if not parts:
                continue

            predicate = parts[0]
            # If the fact is an 'at' predicate with two arguments (object and location):
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            # If the fact is an 'in' predicate with two arguments (package and vehicle):
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle
            # Other predicates (like 'capacity', 'road', etc.) are not needed for this heuristic and are ignored.

        total_cost = 0  # Initialize the total estimated cost.

        # Calculate the estimated cost for each package that needs to reach a specific goal location.
        for package, goal_loc in self.package_goals.items():
            package_cost = 0 # Initialize cost for the current package.

            # Check if the package is currently on the ground at some location.
            if package in current_locations:
                current_loc = current_locations[package]
                # If the package is on the ground at its goal location, it requires no further actions.
                if current_loc == goal_loc:
                    package_cost = 0
                else:
                    # If the package is on the ground at a different location, it needs a full transport cycle:
                    # pick-up (1) + drive (1) + drop (1) = 3 actions.
                    package_cost = 3
            # Check if the package is currently inside a vehicle.
            elif package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find the current location of the vehicle holding the package.
                # Use .get() for safety, although in a valid state, a vehicle should always have an 'at' fact.
                vehicle_loc = current_locations.get(vehicle)

                # If the vehicle's location is known:
                if vehicle_loc is not None:
                    # If the vehicle is at the package's goal location, only a drop action is needed.
                    if vehicle_loc == goal_loc:
                        package_cost = 1 # drop
                    else:
                        # If the vehicle is at a different location, it needs to drive to the goal
                        # and then drop the package.
                        package_cost = 2 # drive + drop
                else:
                    # If the vehicle's location is unknown (unexpected state), assign a penalty.
                    # Estimate a full transport cycle cost.
                    package_cost = 3 # pick (implicitly needed to get into vehicle), drive, drop
            else:
                 # If the package is neither 'at' a location nor 'in' a vehicle (unexpected state),
                 # assign a penalty. Estimate a full transport cycle cost.
                 package_cost = 3 # pick, drive, drop

            # Add the estimated cost for this package to the total cost.
            total_cost += package_cost

        # The total_cost calculated represents the sum of estimated actions for all
        # packages not yet at their goal. This sum is 0 if and only if all goal
        # packages are at their goal locations, which defines the goal state.
        return total_cost
