from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example:
      fact = '(on r1 r2)' --> returns ['on', 'r1', 'r2']
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., "(on r1 r2)".
    - args: The expected pattern (wildcards '*' allowed).
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rod-rings domain.

    Summary:
    This heuristic computes the estimated cost to achieve all goal conditions
    in a rod-rings problem instance by counting how many goal facts are not
    currently satisfied. Each unsatisfied goal fact contributes 1 to the
    estimated cost. The resulting heuristic is not guaranteed to be admissible,
    but it is straightforward to compute and provides a domain-aware estimate.

    Assumptions:
    1. A goal state requires that all facts in the goal subset be true.
    2. The domain includes only the predicates:
       (on r1 r2), (top r s), (bot r s), and (holding r).
    3. Each action put_top or put_bottom rearranges exactly one ring at a time,
       but can satisfy multiple predicates at once.

    Heuristic Initialization:
    - Extract all goal facts from the task (i.e., rods, rings, the "on", "top",
      "bot", and "holding" relations needed at the end).
    - No static facts are used in this domain, so the constructor just keeps
      the goal set.

    Step-By-Step Thinking for Computing Heuristic:
    1. In the call operator, check if the current state already contains all
       goal facts. If yes, return 0.
    2. If not, count how many goal facts are not satisfied by the current
       state (i.e., those not present in the state's fact set).
    3. Return the count of missing goal facts as the heuristic value.

    This simple counting approach ensures the returned value is 0 when the
    goal is reached, and it remains finite for any reachable intermediate
    state. Though it may overestimate the actual cost in terms of actions,
    it is computationally efficient and leverages the domain's goal facts
    to guide the search.
    """

    def __init__(self, task):
        """
        Store the goal conditions for use in the heuristic.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Not used in this domain, but stored for completeness.

    def __call__(self, node):
        """
        Calculate the heuristic value as the number of goal facts not yet satisfied.
        """
        current_state = node.state

        # If all goals are satisfied, return 0.
        if self.goals.issubset(current_state):
            return 0

        # Count how many goal facts are missing from the current state.
        missing_goals = sum(1 for fact in self.goals if fact not in current_state)
        return missing_goals