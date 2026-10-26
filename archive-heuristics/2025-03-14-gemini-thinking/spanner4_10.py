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
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal.
    It counts the minimum actions needed for each nut individually and sums them up.
    The actions considered are walking to the nut's location, picking up a usable spanner if not carrying one,
    and finally tightening the nut.

    # Assumptions:
    - For each nut that needs to be tightened, we assume there is always a usable spanner available,
      either at the current location or at some reachable location.
    - We simplify the cost of getting a spanner if not currently carrying one to a single action count.
      This might underestimate the actual cost in scenarios where spanners are far away, but it maintains efficiency.
    - We assume that for each nut, we need to perform at most one walk action to reach the nut's location,
      one pickup_spanner action to get a usable spanner, and one tighten_nut action to tighten it.

    # Heuristic Initialization
    - Extracts the goal nuts from the task's goal conditions.
    - No static facts are explicitly used in this simplified heuristic, although location links could be used for a more sophisticated walk cost estimation.

    # Step-By-Step Thinking for Computing Heuristic
    For each nut that is specified to be tightened in the goal:
    1. Check if the nut is already tightened in the current state. If yes, no further actions are needed for this nut (cost is 0).
    2. If the nut is not tightened, initialize the estimated cost for this nut to 1 (for the 'tighten_nut' action itself).
    3. Determine the location of the nut from the current state using the 'at' predicate.
    4. Determine the location of the man from the current state using the 'at' predicate.
    5. If the man is not at the same location as the nut, increment the cost by 1 (for a 'walk' action to reach the nut).
    6. Check if the man is currently carrying a usable spanner. Iterate through the state to find predicates 'carrying' and 'usable'.
       If the man is not carrying any usable spanner, increment the cost by 1 (for a 'pickup_spanner' action).
    7. Sum up the estimated costs for all goal nuts. This sum is the total heuristic estimate for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal nuts from the task goals.
        """
        self.goals = task.goals
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """
        Estimate the number of actions required to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0

        for nut_name in self.goal_nuts:
            if f'(tightened {nut_name})' in state:
                continue # Nut already tightened, no cost

            nut_cost = 1 # Cost for tighten_nut action

            nut_location = None
            man_location = None
            man_carrying_usable_spanner = False

            for fact in state:
                if match(fact, "at", nut_name, "*"):
                    nut_location = get_parts(fact)[2]
                elif match(fact, "at", "*", "*") and get_parts(fact)[1] == 'bob': # Assuming man is always 'bob'
                    man_location = get_parts(fact)[2]
                elif match(fact, "carrying", "*", "*") and get_parts(fact)[1] == 'bob': # Assuming man is always 'bob'
                    carried_spanner = get_parts(fact)[2]
                    if f'(usable {carried_spanner})' in state:
                        man_carrying_usable_spanner = True

            if nut_location and man_location and nut_location != man_location:
                nut_cost += 1 # Cost for walk action

            if not man_carrying_usable_spanner:
                nut_cost += 1 # Cost for pickup_spanner action

            heuristic_value += nut_cost

        return heuristic_value
