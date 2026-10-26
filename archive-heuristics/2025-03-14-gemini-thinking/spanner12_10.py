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
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It considers the actions: pickup_spanner, walk, and tighten_nut. For each nut that needs to be tightened,
    it estimates the cost based on whether the man is carrying a usable spanner and is at the nut's location.

    # Assumptions
    - For each nut that needs to be tightened, we assume we need at most one pickup_spanner action,
      at most one walk action, and exactly one tighten_nut action.
    - We assume there is always a usable spanner available and a path to every location if needed.
    - The heuristic is not admissible but aims to be informative and efficient for greedy best-first search.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - No static facts are explicitly used in this heuristic, although link information could be used for more sophisticated path cost estimation if needed.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition of the form '(tightened ?nut)' that is not satisfied in the current state:
    1. Initialize a cost counter for this nut to 0.
    2. Check if the man is currently carrying a usable spanner. If not, increment the cost by 1 (for pickup_spanner action).
    3. Check if the man is at the same location as the nut. If not, increment the cost by 1 (for walk action).
    4. Increment the cost by 1 for the tighten_nut action itself.
    5. Sum up the costs for all goal nuts that are not yet tightened to get the total heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        goal_nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, 'tightened', '*'):
                goal_nuts_to_tighten.add(get_parts(goal)[1])

        tightened_nuts_in_state = set()
        for fact in state:
            if match(fact, 'tightened', '*'):
                tightened_nuts_in_state.add(get_parts(fact)[1])

        unsatisfied_goal_nuts = goal_nuts_to_tighten - tightened_nuts_in_state

        for nut_name in unsatisfied_goal_nuts:
            nut_cost = 0

            carrying_usable_spanner = False
            for fact in state:
                if match(fact, 'carrying', 'bob', '*') and match(fact, 'usable', get_parts(fact)[2]):
                    carrying_usable_spanner = True
                    break
            if not carrying_usable_spanner:
                nut_cost += 1  # pickup_spanner

            man_location = None
            nut_location = None
            for fact in state:
                if match(fact, 'at', 'bob', '*'):
                    man_location = get_parts(fact)[2]
                if match(fact, 'at', nut_name, '*'):
                    nut_location = get_parts(fact)[2]

            if man_location != nut_location:
                nut_cost += 1  # walk

            nut_cost += 1  # tighten_nut
            heuristic_value += nut_cost

        return heuristic_value
