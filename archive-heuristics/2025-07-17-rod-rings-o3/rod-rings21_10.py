from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Return the list of tokens that constitute a PDDL fact.

    Example
    -------
    "(top r1 s1)"  ->  ["top", "r1", "s1"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight domain–dependent heuristic for the **rod-rings** domain.
    It estimates the number of actions that are still necessary to obtain
    every `top`, `bot` and `on` relation required by the goal.

    Assumptions
    -----------
    1.  Goal descriptions only *add* facts (no negative goals).
    2.  Each action (`put_top` / `put_bottom`) can establish **at most one**
        `top`/`bot` fact *and* **at most one** `on` fact.
       Hence weighting an unsatisfied `top`/`bot` fact with *2* and an
       unsatisfied `on` fact with *1* provides a plausible lower bound on
       the number of actions that are still required.
    3.  The predicate `holding` is ignored in the goal because the planner is
        allowed to finish while still holding a ring (the PDDL goals of the
        benchmark sets never restrict `holding`).

    Heuristic Initialization
    ------------------------
    * The constructor stores the set of goal facts for fast set-difference
      tests during the search.
    * No static information is required because the domain has *no* static
      predicates.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1.  Determine the set of currently **unsatisfied** goal facts by
        computing `self.goals - state`.
    2.  For every unsatisfied fact
        - add **2** to the heuristic if it is a `top` or `bot` predicate,
        - add **1** if it is an `on` predicate,
        - add **1** for any other predicate (included for completeness).
    3.  The resulting weighted sum is returned.  
        It is zero *iff* all goal facts are already contained in the state
        (i.e.\ the state is a goal state); it is finite for every reachable
        state because the difference set is finite.
    """

    def __init__(self, task):
        """Store all positive goal literals for quick access."""
        self.goals = task.goals

    def __call__(self, node):
        """Return the heuristic value for the given search node."""
        state = node.state

        # Facts that are required by the goal but not (yet) true in the state
        unsatisfied = self.goals - state

        cost = 0
        for fact in unsatisfied:
            predicate = get_parts(fact)[0]
            if predicate in ("top", "bot"):
                cost += 2
            elif predicate == "on":
                cost += 1
            else:
                cost += 1  # fallback for other predicates (e.g., 'holding')

        return cost
