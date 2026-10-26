from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove surrounding parentheses from a fact string and split it into components.
    For example, '(on r1 r2)' becomes ['on', 'r1', 'r2'].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Example:
        fact = '(on r1 r2)'
        pattern = ('on', '*', 'r2')
    returns True if fact matches the pattern, otherwise False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the rod-rings domain. Estimates the number of
    additional actions needed to achieve all goal conditions.

    Assumptions
    -----------
    - The domain only includes rings and sticks, with relevant predicates on/top/bot/holding.
    - A single action (put_top or put_bottom) can establish or destroy multiple facts.
    - The heuristic does not need to be admissible; it only aims to guide a greedy best-first search.

    Heuristic Initialization
    -------------------------
    - We store the goal conditions (from the task) and any static facts that might be used
      for domain-specific reasoning (though rod-rings does not typically have static relationships).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already satisfied in the current state, return 0.
    2. Otherwise, count how many goal facts are not yet satisfied.
    3. Return this count as an estimate of the number of actions required. This simple approach
       often suffices to guide the search without overcomplicating the computation. Each unsatisfied
       goal relation will likely need at least one put_top or put_bottom action to fix.
    """

    def __init__(self, task):
        """
        Store the goals and static facts for potential domain-dependent reasoning.
        """
        self.goals = task.goals
        self.static = task.static  # Unused for now, but available if needed.

    def __call__(self, node):
        """
        Compute the heuristic by counting how many goal facts are not satisfied in the current state.
        """
        state = node.state

        # Check if it's already a goal state.
        if all(goal in state for goal in self.goals):
            return 0

        # Count how many goal facts are unsatisfied.
        missing_goals = sum(1 for g in self.goals if g not in state)

        return missing_goals