from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid state representations,
        # but we handle it defensively.
        return [] # Return empty list for malformed facts

    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, summing the estimates for
    all packages that are not yet at their goal. It provides a lower bound
    on the number of actions needed for each individual package, assuming
    vehicles are available.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost for a package not at its goal is estimated based on its current
      state (on the ground or in a vehicle) relative to the goal location.
    - This heuristic ignores vehicle capacity constraints and the actual road
      network distance, treating any required drive as a single unit of cost.
    - It sums the costs for each package independently, which might overestimate
      or underestimate the true cost in scenarios involving shared vehicle trips.
      However, for greedy search, this simple estimate can be effective.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each package specified in the goal:
       a. Determine the package's goal location from the initialized goal mapping (`self.package_goals`).
       b. Find the package's current location in the current state. This involves iterating
          through the state facts to find either an `(at package location)` fact or
          an `(in package vehicle)` fact.
       c. If the package is currently located at its goal location on the ground
          (i.e., the fact `(at package goal_location)` is present in the state),
          the cost for this package is 0. Continue to the next package.
       d. If the package is not at its goal location on the ground:
          i. Determine if the package is currently on the ground or inside a vehicle.
             This is done by checking if there is an `(in package vehicle)` fact
             for this package in the state.
          ii. If the package is on the ground at `current_loc` (and `current_loc` is not the goal):
              - It needs to be picked up (1 action).
              - It needs to be transported by a vehicle (estimated as 1 drive action).
              - It needs to be dropped at the goal (1 action).
              - Estimated cost for this package: 3 actions (Pick + Drive + Drop).
          iii. If the package is inside a vehicle `v`:
              - Find the current location of vehicle `v` by looking for an `(at vehicle location)` fact in the state.
              - If vehicle `v` is currently at the package's goal location:
                - The package needs to be dropped (1 action).
                - Estimated cost for this package: 1 action (Drop).
              - If vehicle `v` is NOT currently at the package's goal location:
                - Vehicle `v` needs to drive to the goal location (estimated as 1 drive action).
                - The package needs to be dropped at the goal (1 action).
                - Estimated cost for this package: 2 actions (Drive + Drop).
    3. The total heuristic value is the sum of the estimated costs for all
       packages that are not yet at their goal location on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each package.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            # We need to parse the goal fact string
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
            # Assuming task.goals provides the set of facts that must be true.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the current state is a goal state. If so, heuristic is 0.
        # This is important for admissibility (though not strictly required for GBFS).
        # The goal is a set of facts. We check if all goal facts are in the state.
        if self.goals <= state:
             return 0

        # If not a goal state, estimate cost.
        total_cost = 0  # Initialize action cost counter.

        # Build current location mappings from the state for quick lookup
        current_locations = {} # Maps object name (package or vehicle) to its location (location name or vehicle name)
        packages_in_vehicles = set() # Set of package names currently inside a vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts and parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                current_locations[package] = vehicle # Package is "located" inside the vehicle
                packages_in_vehicles.add(package)

        # Iterate through the packages we care about (those in the goal)
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground.
            # The goal is (at package goal_location).
            goal_fact_for_package = f"(at {package} {goal_location})"
            if goal_fact_for_package in state:
                # Package is already at its goal location on the ground. Cost for this package is 0.
                continue

            # Package is not at its goal location on the ground.
            # It's either on the ground elsewhere or inside a vehicle.

            # Find the package's current whereabouts (location or vehicle)
            current_package_whereabouts = current_locations.get(package)

            # If the package is not found in the state's location facts, something is wrong
            # or it's not relevant to the goal (shouldn't happen for packages in goals).
            if current_package_whereabouts is None:
                 # This package is not in the state in an 'at' or 'in' predicate.
                 # This indicates an issue with the state representation or problem definition.
                 # Assigning a cost of 3 (like being on the ground elsewhere) seems reasonable
                 # as it needs pickup, transport, and drop.
                 total_cost += 3
                 continue


            # Case 1: Package is on the ground at current_package_whereabouts
            # This happens if the package name is NOT in the packages_in_vehicles set.
            if package not in packages_in_vehicles:
                 # Package is on the ground at current_package_whereabouts (which must be a location string)
                 # Needs: Pick-up + Drive + Drop
                 # Simplified cost: 3 actions
                 total_cost += 3
            # Case 2: Package is inside a vehicle (package name IS in packages_in_vehicles set)
            else:
                 vehicle_name = current_package_whereabouts # This is the vehicle name
                 vehicle_location = current_locations.get(vehicle_name) # Where is the vehicle?

                 # If the vehicle the package is in is not found in the state's location facts, something is wrong.
                 if vehicle_location is None:
                     # The vehicle is not located anywhere. This is an invalid state.
                     # Assigning a cost of 2 (Drive + Drop) seems reasonable as the package is in a vehicle,
                     # assuming the vehicle can eventually get to the goal.
                     total_cost += 2
                     continue

                 if vehicle_location == goal_location:
                     # Vehicle is at the goal location. Needs: Drop
                     # Simplified cost: 1 action
                     total_cost += 1
                 else:
                     # Vehicle is not at the goal location. Needs: Drive + Drop
                     # Simplified cost: 2 actions
                     total_cost += 2

        # The heuristic value is the sum of estimated costs for all packages
        # that are not yet at their goal location on the ground.
        return total_cost
