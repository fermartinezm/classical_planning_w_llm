from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses
    and splitting into individual tokens.

    Example:
    If fact = "(holding r1)", get_parts(fact) -> ["holding", "r1"].
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This is a domain-dependent heuristic for the rod-rings domain. It provides a
    simple cost estimate of how many additional actions are needed to reach the goal
    state from the current state.

    Assumptions
    -----------
    • Each goal condition in rod-rings typically relates to rings being "on" each other
      or being at the "top" or "bottom" of a stick.  
    • We assume that satisfying each goal fact requires at least one action, though
      in practice one action can fix multiple goal facts.

    Heuristic Initialization
    ------------------------
    • The constructor records the subset of facts that must be satisfied for the
      goal state (the rod-rings domain does not rely on static facts, but we store
      them for completeness).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if all goal facts are already satisfied:
       - If yes, the heuristic returns 0 because we are already in a goal state.
    2. Otherwise, count the number of goal facts not yet satisfied in the current state.
    3. Return that count as the heuristic value. Although simplistic, this domain-dependent
       heuristic at least checks each rod-ring specific goal condition rather than 
       ignoring domain structure altogether.
    """

    def __init__(self, task):
        """
        Initialize the rod_ringsHeuristic.

        :param task: The planning task, containing:
            - task.goals: A set of goal facts.
            - task.static: A set of static facts (unused here).
        """
        self.goals = task.goals
        self.static = task.static  # Not strictly necessary for this heuristic.

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        We return the number of unsatisfied goal facts. This yields 0 exactly when
        the current state already satisfies all goals.

        :param node: A search node whose .state is a frozenset of PDDL facts.
        :return: An integer representing the estimated cost to reach the goal.
        """
        # If the state's facts include all goal facts, return 0.
        if self.goals.issubset(node.state):
            return 0

        # Otherwise, count how many goal facts are not in the current state.
        unsatisfied_goals = self.goals - node.state
        return len(unsatisfied_goals)