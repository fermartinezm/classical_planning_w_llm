from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe log a warning or raise an error
        # For robustness, return empty list or handle gracefully
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions required to get each
    package to its goal location, summing the estimates for all packages. It
    ignores vehicle capacity constraints and the road network structure (distance).

    # Assumptions
    - Actions have unit cost.
    - Each package needs a sequence of actions to reach its goal if it's not already there.
    - If a package is on the ground at a location different from its goal, it needs
      at least 3 actions: pick-up, drive (by a vehicle), and drop.
    - If a package is inside a vehicle, it needs at least 2 actions: drive (by the vehicle)
      to the goal location and drop.
    - This heuristic is admissible (never overestimates the true cost) because it
      ignores positive interactions (like carrying multiple packages or vehicles sharing roads)
      and resource constraints (capacity).

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goal conditions.
    - Static facts (like `road` and `capacity-predecessor`) are available but not
      used by this simple heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the goal location for every package that needs to be transported
       (these are the packages mentioned in the goal state).
    2. For each such package, determine its current status in the given state:
       - Is it on the ground at a specific location `(at ?p ?l)`?
       - Is it inside a vehicle `(in ?p ?v)`?
    3. Iterate through each package that has a goal location:
       - If the package is currently at its goal location `(at ?p ?l_goal)`:
         The cost contribution for this package is 0.
       - If the package is currently on the ground at a location `l_current`
         that is *not* its goal location (`(at ?p ?l_current)` where `l_current != l_goal`):
         It needs to be picked up, transported (drive), and dropped. This requires
         at least 3 actions. Add 3 to the total heuristic cost.
       - If the package is currently inside a vehicle (`(in ?p ?v)`):
         It needs to be transported (drive by the vehicle) and dropped. This requires
         at least 2 actions. Add 2 to the total heuristic cost.
    4. The total heuristic value is the sum of the costs calculated for each package.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are available in task.static but not used by this heuristic.
        # task.static = frozenset({...})

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # We ignore other potential goal predicates if any exist, as 'at' is the primary goal for packages.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Map packages to their current status (at location or in vehicle)
        package_current_status = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                # (at locatable location) - could be package or vehicle
                # We only care about packages here. We need a way to distinguish packages from vehicles.
                # A simple way is to check if the object is in our goal_locations keys.
                obj, location = parts[1], parts[2]
                if obj in self.goal_locations: # Assume objects in goal_locations are packages
                     package_current_status[obj] = ('at', location)
            elif predicate == "in" and len(parts) == 3:
                # (in package vehicle)
                package, vehicle = parts[1], parts[2]
                if package in self.goal_locations: # Assume objects in goal_locations are packages
                    package_current_status[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package's current status is known and relevant
            if package in package_current_status:
                status, loc_or_veh = package_current_status[package]

                # Case 1: Package is at its goal location
                if status == 'at' and loc_or_veh == goal_location:
                    # Cost for this package is 0, it's already where it needs to be.
                    pass
                # Case 2: Package is on the ground, but not at its goal location
                elif status == 'at' and loc_or_veh != goal_location:
                    # Needs pick-up, drive, drop. Minimum 3 actions.
                    total_cost += 3
                # Case 3: Package is inside a vehicle
                elif status == 'in':
                    # Needs drive (by vehicle) to goal location and drop. Minimum 2 actions.
                    total_cost += 2
                # else: Should not happen in a valid state if package_current_status is complete
                # for all packages in goal_locations.

            # else: Package is in goal_locations but not found in state facts (neither 'at' nor 'in').
            # This indicates an issue with the state representation or problem definition,
            # or the package is not relevant to the current state (e.g., already achieved its goal
            # and was removed from state facts, which is not standard STRIPS).
            # Assuming standard STRIPS where all true facts are listed, this case implies
            # the package is not in the state, which shouldn't happen for a package
            # that needs to reach a goal location. We proceed assuming packages in goal_locations
            # will be found in the state facts if they are not yet at their goal.

        return total_cost

