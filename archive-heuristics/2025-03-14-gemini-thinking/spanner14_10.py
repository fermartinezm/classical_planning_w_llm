from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the preconditions for the 'tighten_nut' action and estimates the cost
    based on the missing preconditions in the current state for each goal nut.

    # Assumptions:
    - The goal is always to tighten a set of nuts.
    - The heuristic assumes that for each loose nut in the goal, we need to perform
      at least the 'tighten_nut' action, and potentially 'walk' and 'pickup_spanner'
      actions to satisfy the preconditions of 'tighten_nut'.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
      It focuses on goals of the form '(tightened ?nut)'.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition '(tightened ?nut)':
    1. Check if the goal condition is already satisfied in the current state. If yes, the cost for this goal is 0.
    2. If the goal is not satisfied, estimate the cost to achieve it by considering the preconditions of the 'tighten_nut' action:
       a. 'carrying ?m - man ?s - spanner' and 'usable ?s - spanner': Check if the man is carrying any usable spanner. If not, assume we need one 'pickup_spanner' action.
       b. 'at ?m - man ?l - location' and 'at ?n - nut ?l - location': Check if the man and the nut are at the same location. If not, assume we need one 'walk' action.
       c. 'loose ?n - nut': This is implicitly assumed if '(tightened ?n)' is not in the state and is a goal. We always need one 'tighten_nut' action.
    3. Sum up the estimated costs for all unsatisfied goal conditions. This sum is the heuristic value for the current state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        for goal_fact in self.goals:
            if goal_fact in state:
                continue  # Goal already achieved, no cost for this goal

            nut_name = get_parts(goal_fact)[1] # Assuming goal is always (tightened ?nut)
            nut_cost = 0

            # Check if carrying a usable spanner
            has_usable_spanner = False
            for fact in state:
                if match(fact, "carrying", "*", "*"):
                    spanner_name = get_parts(fact)[2]
                    if f"(usable {spanner_name})" in state:
                        has_usable_spanner = True
                        break
            if not has_usable_spanner:
                nut_cost += 1  # Estimate cost for pickup_spanner

            # Check if man and nut are at the same location
            man_at_nut_location = False
            man_location = None
            nut_location = None

            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj_name = get_parts(fact)[1]
                    location_name = get_parts(fact)[2]
                    if match(fact, "at", "bob", "*"): # Assuming man's name is bob
                        man_location = location_name
                    if obj_name == nut_name:
                        nut_location = location_name

            if man_location != nut_location:
                nut_cost += 1  # Estimate cost for walk

            nut_cost += 1  # Estimate cost for tighten_nut
            heuristic_value += nut_cost

        return heuristic_value
