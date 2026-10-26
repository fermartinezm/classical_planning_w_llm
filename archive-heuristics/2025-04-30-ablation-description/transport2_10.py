# Need to import fnmatch for pattern matching on fact strings
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment if using a base class

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at package1 location1)" -> ["at", "package1", "location1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    # fnmatch handles '*' correctly, but we should at least have the same number of elements
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic: # Inherit from Heuristic if using a base class
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated
    cost for each package that is not yet at its goal location. The estimate for
    a single package depends on whether it is currently on the ground or inside a vehicle.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost for a package on the ground not at its goal is estimated as 3 actions
      (pick-up, drive, drop). This assumes a vehicle is available and a single drive
      action is sufficient for the required transport.
    - The cost for a package inside a vehicle not at its goal is estimated as 2 actions
      (drive, drop). This assumes the vehicle can reach the goal location in a single
      drive action.
    - This heuristic ignores vehicle capacity constraints and the actual distance
      or road network structure, simplifying the problem to independent package movements.
    - The heuristic is 0 if and only if all goal conditions related to package locations are met.

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task's goal conditions.
    - Static facts (like road network or capacity-predecessors) are noted but not
      explicitly used in the current heuristic calculation, as it abstracts away
      the complexities of vehicle movement and capacity management.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the goal location for each package from the task's goal conditions.
       Store this mapping (package -> goal_location). Only consider goals of the form `(at ?p ?l)`.
    3. Iterate through each package identified in the goals.
    4. For the current package `p` and its goal location `l_goal`:
       a. Check if the fact `(at p l_goal)` exists in the current state.
          If it exists, the package is already at its goal. Add 0 to the total cost for this package.
       b. If `(at p l_goal)` does not exist in the state, the package is not at its goal.
          Determine the package's current status by searching the state:
          - Look for a fact `(at p l_current)` where `l_current` is any location.
            If found, the package is on the ground at `l_current`. Since `l_current` must be different from `l_goal` (otherwise step 4a would have applied), this package needs to be picked up, transported, and dropped. Add 3 to the total cost for this package.
          - If not found on the ground, look for a fact `(in p v)` where `v` is any vehicle.
            If found, the package is inside vehicle `v`. This package needs to be transported (by the vehicle) and dropped. Add 2 to the total cost for this package.
          - Assume that for any package mentioned in the goals, it will be found either `at` a location or `in` a vehicle in any valid state.
    5. The total heuristic value is the sum accumulated from all packages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages.
        """
        self.goals = task.goals
        # self.static_facts = task.static # Static facts are available but not used in this simple heuristic

        # Store goal locations for each package.
        # Assuming goals are only of the form (at package location)
        self.package_goal_locations = {}
        for goal in self.goals:
            # Example goal: '(at p1 l2)'
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's an (at ?p ?l) fact
                package, location = args
                # Only consider packages (type 'package') for goals
                # We don't have type information here, rely on the fact structure
                # Assuming the first argument of (at) in a goal is always a package
                self.package_goal_locations[package] = location
            # Ignore other potential goal types or malformed goals

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package whose goal location is specified
        for package, goal_location in self.package_goal_locations.items():
            # Check if the package is already at its goal location
            goal_fact = f"(at {package} {goal_location})"
            if goal_fact in state:
                # Package is at its goal, no cost added for this package
                continue

            # Package is not at its goal. Find its current status.
            # It must be either 'at' some other location or 'in' a vehicle.
            is_on_ground = False
            is_in_vehicle = False

            # Check if the package is on the ground at some location
            # Iterate through the state to find a fact like (at package ?)
            for fact in state:
                # Use match for robustness, though direct string check might be faster if format is guaranteed
                if match(fact, "at", package, "*"):
                    # Found the package on the ground at some location
                    is_on_ground = True
                    break # Found its location, no need to search further for 'at'

            # If not found on the ground (and not at goal), it must be in a vehicle
            if not is_on_ground:
                 for fact in state:
                     # Use match for robustness
                     if match(fact, "in", package, "*"):
                         # Found the package inside a vehicle
                         is_in_vehicle = True
                         break # Found its vehicle, no need to search further for 'in'

            # Calculate cost based on current status
            if is_on_ground:
                # Package is on the ground, not at goal. Needs pick-up, drive, drop.
                total_cost += 3
            elif is_in_vehicle:
                # Package is in a vehicle, not at goal. Needs drive, drop.
                total_cost += 2
            # Else: This case should ideally not happen in a valid state for a package
            # mentioned in the goals. If it did, it would add 0, which is incorrect.
            # We assume valid states where packages are always locatable.

        return total_cost
