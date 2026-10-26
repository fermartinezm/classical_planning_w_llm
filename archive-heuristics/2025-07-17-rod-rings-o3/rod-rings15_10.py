import math
from heuristics.heuristic_base import Heuristic


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A very light-weight domain–dependent heuristic for the *rod-rings* domain.
    The heuristic exploits the fact that a single action (`put_top` or
    `put_bottom`) can simultaneously establish **at most two** goal facts:
    - one `top` fact (the new top ring on a stick) **and**
    - one `on` fact (the newly placed ring being on another ring).

    By counting how many goal facts are still unsatisfied and dividing this
    number by two (rounding up), we obtain a lower-bound on how many actions
    are still necessary.  
    The heuristic is inexpensive (𝑂(|goals|)) and sufficiently informative to
    guide a greedy best-first search while never over-estimating the number of
    remaining actions too drastically.

    Assumptions
    -----------
    1. Every *rod-rings* operator affects only one stick.
    2. Each operator can achieve **at most two** of the goal literals
       simultaneously (`top` + `on`).
    3. Whether a particular ring is currently held is irrelevant for the goal.

    Heuristic Initialization
    ------------------------
    Only the set of goal literals has to be cached.  
    Static information is empty for the domain, so no additional
    preprocessing is required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Take the current state `s` supplied by the search node.
    2. Iterate over every goal literal `g` in `self.goals`
       and count how many of them are **not** contained in `s`.
       This gives `unsat`, the number of unsatisfied goal facts.
    3. Because one action can make at most two of those facts true,
       the *minimum* number of additional actions is the ceiling of
       `unsat / 2`.
    4. Return that number as the heuristic value.
       • Goal states yield `unsat = 0` ⇒ heuristic value `0`.  
       • For every other state we return a finite, positive integer.
    """

    def __init__(self, task):
        # Cache the (immutable) goal set for quick membership tests.
        self.goals = task.goals

    def __call__(self, node):
        """
        Compute `ceil(#unsatisfied_goal_literals / 2)`.
        """
        state = node.state

        # Count how many goal facts are still missing in the current state.
        unsatisfied = sum(1 for g in self.goals if g not in state)

        # Each action can satisfy at most two goal facts, hence the ceiling.
        return math.ceil(unsatisfied / 2)
