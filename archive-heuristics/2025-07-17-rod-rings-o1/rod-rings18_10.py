from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This domain-dependent heuristic for the rod-rings domain estimates the
    number of actions required to satisfy all goal facts by counting how many
    of them are currently unsatisfied. Each unsatisfied goal fact is assumed
    to require (at least) one additional move to fix.

    Assumptions
    -----------
    1. Any goal fact that is not present in the current state contributes a
       cost of 1.
    2. A single action (put_top or put_bottom) can potentially fix exactly one
       goal fact among (on ?top ?bot), (top ?r ?s), (bot ?r ?s), or (holding ?r).
    3. The domain does not require any extra cost for “picking up a ring”
       or “unholding” beyond the single action that swaps rings, since the
       domain’s actions encapsulate both picking up and placing within a
       single operator.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal facts directly from the task, along with any
      static facts. For rod-rings, static facts are often empty or unused,
      but they are stored for completeness.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which of the goal facts are currently unsatisfied by comparing
       the current state’s set of facts with the set of goal facts.
    2. The heuristic value is the number of these unsatisfied goal facts.
       - If there are no unsatisfied goal facts, the heuristic is 0.
       - Otherwise, it is equal to the count of all unsatisfied facts, providing
         an estimate of how many swaps are needed.
    3. Return this computed value as the heuristic estimate.

    Note
    ----
    Although this heuristic is “domain-dependent,” it is not guaranteed to be
    admissible. It is designed to be simple yet effective enough to reduce the
    number of expanded nodes in a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the rod-rings domain heuristic by recording the task goals
        and static facts.
        """
        self.goals = task.goals
        self.static = task.static  # Not used in this heuristic but kept for completeness

    def __call__(self, node):
        """
        Compute and return the heuristic value for the given state.

        Parameters
        ----------
        node : search.Node
            The node whose state's heuristic value is to be calculated.

        Returns
        -------
        int
            The estimated cost (in number of actions) to reach a goal state
            from this node's state.
        """
        current_state = node.state
        # Count how many goal facts are not yet satisfied.
        unsatisfied = self.goals - current_state
        return len(unsatisfied)