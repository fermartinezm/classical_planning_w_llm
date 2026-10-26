import math
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Helper that removes the surrounding parentheses of a PDDL fact
    and splits it into its individual components.
    Example: "(top r2 s1)"  ->  ["top", "r2", "s1"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    Fast, domain-specific heuristic for the *rod-rings* domain.  The
    heuristic estimates how many `put_top` / `put_bottom` actions are still
    required until all goal facts that refer to `(top …)` and `(on … …)`
    are satisfied.

    Assumptions
    -----------
    1. Only the predicates `top` and `on` occur in goal descriptions.
    2. One operator can establish **at most two** goal facts (one `top`
       and one `on`).  Therefore we charge separately for every still
       unsatisfied goal fact.
    3. “Accessibility” of a ring strongly influences the cost of moving
       it:
       • 0 if the ring is already **held**.  
       • 1 if the ring is currently at a **stick’s extremity**
         (`top` *or* `bot`).  
       • 2 otherwise (buried somewhere inside a stack).

    Heuristic Initialization
    ------------------------
    The constructor stores
        – the complete set of goal facts (`task.goals`)
    No static information exists in this domain, hence nothing else has to
    be cached.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node we
    1. Build three quick-lookup tables from the current state  
       `held_rings`, `top_rings`, `bot_rings`.
    2. Define `access_cost(r)` exactly as described in *Assumptions*.
    3. Identify all still **unsatisfied** goal facts  
       `unsat = goals − state`.
    4. For every unsatisfied goal fact add a cost estimate:
       • `(top R S)`   →  1 + access_cost(R)  
         (one action to place R, plus effort to get hold of it)
       • `(on A B)`    →  1 + access_cost(A) + access_cost(B)  
         (we must be able to manipulate *both* rings and then perform
         one placement action)
    5. Return the sum.  By construction the heuristic is
       – 0 only in a goal state,  
       – strictly positive and finite in every other reachable state.
    """

    def __init__(self, task):
        # Goals are needed every time the heuristic is called
        self.goals = task.goals

    # ------------------------------------------------------------------ #
    # Main heuristic evaluation                                           #
    # ------------------------------------------------------------------ #
    def __call__(self, node):
        state = node.state

        # 1)  Fast exit if we already reached the goal
        unsatisfied = self.goals - state
        if not unsatisfied:
            return 0

        # 2)  Extract where each ring currently is
        held_rings = set()
        top_rings = set()
        bot_rings = set()

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "holding":
                held_rings.add(args[0])          # (holding R)
            elif pred == "top":
                top_rings.add(args[0])           # (top R S)
            elif pred == "bot":
                bot_rings.add(args[0])           # (bot R S)

        def access_cost(ring: str) -> int:
            """Return how “hard” it is to get direct control over <ring>."""
            if ring in held_rings:
                return 0
            if ring in top_rings or ring in bot_rings:
                return 1
            return 2  # somewhere in the middle of a stack

        # 3)  Accumulate the estimated effort for every unsatisfied goal fact
        h_value = 0
        for fact in unsatisfied:
            pred, *args = get_parts(fact)

            if pred == "top":
                ring = args[0]
                h_value += 1 + access_cost(ring)

            elif pred == "on":
                upper, lower = args
                h_value += 1 + access_cost(upper) + access_cost(lower)

            # No other predicate can be part of a goal expression
            # in this domain, so we silently ignore different ones.

        return h_value
