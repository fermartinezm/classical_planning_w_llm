from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting.
    Example: "(holding r1)" -> ["holding", "r1"].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (wildcards allowed).
    Example: match("(on r1 r2)", "on", "*", "*") -> True.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic provides an estimate of how many actions are needed to
    achieve all goal conditions in the rod-rings domain. It sums the number
    of unsatisfied goal facts in the current state, assuming each unsatisfied
    fact will likely require at least one action to fix.

    Assumptions
    -----------
    1. Only one ring can be held at a time (as per the 'holding ?r' predicate).
    2. Each action (put_top or put_bottom) can resolve at least one missing
       goal fact.
    3. The heuristic does not need to be admissible, only to reduce search
       expansions.

    Heuristic Initialization
    -------------------------
    - The constructor stores all goal facts from the given task in self.goals.
    - Static facts (if any) are also available in the task, but the basic
      counting approach here does not specifically rely on them.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Read the current state from the node's state (a frozenset of facts).
    2. Check each goal fact against the current state to see if it is satisfied.
    3. Count how many goal facts are currently unsatisfied.
    4. Return that count as the heuristic value:
       - If all goals are satisfied, the heuristic is 0.
       - Otherwise, the heuristic is the number of goal facts missing in the
         current state.
    """

    def __init__(self, task):
        """
        Store the goal conditions and any static facts from the task
        for potential use in more advanced heuristics.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Not used here, but stored if needed.

    def __call__(self, node):
        """
        Return the estimated cost to reach a goal state by counting how many
        goal facts are unsatisfied in the current state.
        """
        state = node.state
        # Count unsatisfied goal facts.
        missing_goals = sum(1 for g in self.goals if g not in state)
        return missing_goals 