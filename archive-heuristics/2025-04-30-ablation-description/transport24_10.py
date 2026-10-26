# Assuming heuristics.heuristic_base exists and contains a Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper functions from examples
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    fact = fact.strip()
    # Check if it starts and ends with parentheses
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    else:
        # This case should ideally not happen with valid PDDL facts in the state/goals
        # print(f"Warning: Fact '{fact}' does not have expected PDDL format.")
        return fact.split() # Fallback

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost by counting the number of packages
    that are not currently at their specified goal location.

    # Assumptions
    - The goal is defined by a set of (at ?p ?l) facts for packages.
    - Each package needs to reach a specific location.
    - The heuristic ignores vehicle capacity, vehicle location, and road network distances.
    - Each action can satisfy at most one (at ?p ?l) goal fact (specifically, a drop action).

    # Heuristic Initialization
    - Extracts the goal locations for each package from the task's goal conditions.
    - Stores these goal locations in a dictionary mapping package names to their target locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for misplaced packages to 0.
    2. Iterate through each package and its corresponding goal location stored during initialization.
    3. For the current package and its goal location, check if the fact `(at package goal_location)` exists in the current state.
       This check is done by constructing the exact string representation of the goal fact and checking for its presence in the state set.
    4. If the fact `(at package goal_location)` is NOT present in the current state, increment the counter.
    5. After checking all packages with goal locations, the total count represents the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages.
        """
        # Assuming task object has a 'goals' attribute which is a set of goal fact strings.
        self.goals = task.goals

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # We are looking for goal facts like "(at package location)"
            parts = get_parts(goal)
            # Check if the fact is an 'at' predicate with 2 arguments after the predicate name
            if parts and parts[0] == "at" and len(parts) == 3:
                 # Assuming the first argument of 'at' in a goal is always a package
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
            # Ignore other types of goal facts if any.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        This heuristic counts the number of packages not at their goal location.
        """
        # Assuming node object has a 'state' attribute which is a frozenset of fact strings.
        state = node.state

        misplaced_packages_count = 0

        # Iterate through each package and its target location from the goals.
        for package, goal_location in self.package_goals.items():
            # Check if the package is currently at its goal location.
            # The state contains facts as strings, e.g., '(at p1 l2)'
            goal_fact_string = f"(at {package} {goal_location})"

            if goal_fact_string not in state:
                misplaced_packages_count += 1

        return misplaced_packages_count
