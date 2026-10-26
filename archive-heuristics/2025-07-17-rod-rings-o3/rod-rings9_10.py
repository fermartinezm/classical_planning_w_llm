from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def split_fact(fact):
    """
    Split a PDDL fact into its components.

    Example
    -------
    "(top r1 s1)"  →  ["top", "r1", "s1"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A simple, fast heuristic for the *rod-rings* domain that counts the weighted
    number of goal facts that are still unsatisfied in the current state.

    Assumptions
    -----------
    1. Only the predicates `top`, `bot` and `on` will occur in goal
       specifications (a `holding` fact is extremely unlikely to appear in a goal
       and is therefore ignored).
    2. Every action can satisfy **at most one** of the following at once  
       – the correct `top` ring, the correct `bot` ring **or** a single `on`
       relation – while potentially destroying others.  
       Consequently we use different weights for the three predicate types:
         • `on`   : 1 action (the relation is introduced directly)  
         • `top`  : 2 actions (ring has to be fetched first, then placed)  
         • `bot`  : 2 actions (analogous to `top`)

    Heuristic Initialization
    ------------------------
    The constructor stores
      • the complete set of goal facts (`task.goals`)
      • a mapping from every goal fact to its pre-computed weight.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Iterate over all goal facts.
    2.  For every goal fact *g* that is **not** contained in the current state
        add `weight(g)` to the running total.
    3.  Return the accumulated sum.

    The heuristic is:
      • `0`  iff **all** goal facts are already true (goal state),
      • guaranteed to be finite for every reachable state,
      • inexpensive – linear in the number of goal facts and independent
        of the overall state size.
    """

    WEIGHTS = {
        "on": 1,
        "top": 2,
        "bot": 2,
        # If a goal should ever contain a `holding` fact we treat it like TOP.
        "holding": 2,
    }

    def __init__(self, task):
        # Store goal facts exactly once for quick subset checks.
        self.goals = frozenset(task.goals)

        # Pre-compute the weight for every goal fact.
        self.goal_weights = {}
        for g in self.goals:
            pred = split_fact(g)[0]
            self.goal_weights[g] = self.WEIGHTS.get(pred, 1)   # default weight 1

    def __call__(self, node):
        """
        Compute the heuristic value for *node.state*.
        """
        state = node.state

        # Fast exit – the most common call once the search reaches the goal.
        if self.goals <= state:
            return 0

        cost = 0
        for g in self.goals:
            if g not in state:
                cost += self.goal_weights[g]

        return cost
