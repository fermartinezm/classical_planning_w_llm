from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
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
    This heuristic estimates the cost by summing up the required operations (pick-up, drop)
    for each package that is not yet at its final destination on the ground.

    # Assumptions
    - The primary goal is to get packages to their specified locations on the ground.
    - Vehicle capacity and specific vehicle locations/paths are ignored for simplicity.
    - The cost of a 'drive' action is implicitly ignored or assumed to be negligible compared to pick/drop operations in this simple heuristic.

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For each package specified in the goal:
    1. Identify the package's goal location.
    2. Check if the package is currently at its goal location on the ground (i.e., the fact `(at package goal_location)` exists in the state).
       - If yes, the cost contribution for this package is 0.
    3. If the package is not at its goal location on the ground:
       - Check if the package is currently inside any vehicle (i.e., the fact `(in package vehicle)` exists for some vehicle).
         - If yes (package is in a vehicle), the cost contribution for this package is 1 (it needs to be dropped at the goal location).
         - If no (package is on the ground at a location other than the goal), the cost contribution for this package is 2 (it needs to be picked up and then dropped at the goal location).
    4. The total heuristic value is the sum of the cost contributions for all packages in the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for each package.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations for each package.
        # Assuming goals are always a conjunction of (at package location) facts.
        self.goal_locations = {}
        # The task.goals object is a frozenset of goal facts.
        for goal_fact in self.goals:
             # Check if the goal fact is an 'at' predicate with two arguments (package, location)
            parts = get_parts(goal_fact)
            if parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Note: This heuristic assumes goals are only (at ?p ?l).
            # More complex goals (e.g., (not (at ?p ?l)), (and ...)) would require more sophisticated parsing.
            # Given the example problems, (at ?p ?l) conjunctions are typical.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package whose location is specified in the goal
        for package, goal_location in self.goal_locations.items():
            # Check if the package is currently at its goal location on the ground
            is_at_goal_on_ground = f"(at {package} {goal_location})" in state

            if is_at_goal_on_ground:
                # Package is already at the goal location on the ground. No cost contribution from this package.
                pass
            else:
                # Package is not at the goal location on the ground.
                # Check if the package is currently inside any vehicle.
                # We need to iterate through the state to find if any fact matches "(in package ?vehicle)"
                is_in_vehicle = False
                for fact in state:
                    if match(fact, "in", package, "*"):
                        is_in_vehicle = True
                        break # Found it, no need to check other facts for this package

                if is_in_vehicle:
                    # Package is in a vehicle. It needs to be dropped at the goal location.
                    # This requires the vehicle to be at the goal location and then a drop action.
                    # We count the drop action (cost 1) and ignore the drive cost for simplicity.
                    total_cost += 1
                else:
                    # Package is on the ground at a location other than the goal location.
                    # It needs to be picked up by a vehicle and then dropped at the goal location.
                    # This requires a vehicle to reach the package, a pick-up action,
                    # the vehicle to drive to the goal, and a drop action.
                    # We count the pick-up (cost 1) and the drop (cost 1) actions, total 2.
                    # We ignore the vehicle movement costs and capacity constraints.
                    total_cost += 2

        return total_cost
