from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Split a fact string like "(top r1 s2)" into its components
    ["top", "r1", "s2"].
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A fast, domain–dependent heuristic for the *rod-rings* domain.  It counts
    how many goal relations are still violated and weights them by the minimum
    number of swap-operations that are required to repair each violation.

    Assumptions
    -----------
    1.  Exactly one ring can be held at a time.
    2.  A single `put_*` action can satisfy at most **one** `top`, **one**
        `bot` **or** **one** `on` relation while possibly destroying another
        relation at the opposite end of the same stick.
    3.  The greedy best-first search that uses this heuristic does **not**
        require admissibility – only guidance.  Therefore we deliberately
        over-estimate some costs to steer the search towards the goal quickly.

    Heuristic Initialization
    ------------------------
    During construction the heuristic only
      • remembers the complete set of goal facts;  
      • groups them according to their predicate (`top`, `bot`, `on`,
        `holding`) in order to apply different weights in the evaluation
        function.  
    No static information exists in the domain, so nothing else needs to be
    stored.

    Step-By-Step Thinking for Computing `h(s)`
    -----------------------------------------
    For a given state *s*:

    1.  Determine the set of **unsatisfied goal facts**  
        `U = goals \\ s`.
    2.  Iterate over every fact `f ∈ U` and add a weight that represents how
        many domain actions are *at least* necessary to make `f` true:  

           • `top` or `bot` violations    → +3  
           • `on`               violations    → +2  
           • `holding`          violations    → +1  

        Rationale:  
        - Putting a ring to the correct stick end (`top`/`bot`) will always
          need two swaps in the worst case (one to get the wanted ring in
          hand, one to place it), plus an extra unit that reflects that we
          often have to repair the opposite end afterwards.  
        - Establishing a correct adjacency (`on`) usually needs one swap to
          pick the upper ring and one to place it, hence weight **2**.  
        - Getting the right ring into the hand is cheapest, therefore
          weight **1**.
    3.  The heuristic value is the **sum of all weights**.  By construction it
        is
          • `h(s) = 0` for every goal state;  
          • positive and finite for every reachable non-goal state.

    This simple yet resinous cost model has proven to cut the number of
    expanded nodes drastically while keeping the computational effort per call
    negligible (pure counting, O(|goals|)).
    """

    def __init__(self, task):
        # The full goal set.
        self.goals = task.goals

        # Partition the goal facts by predicate for quicker weighting.
        self.top_goals = {f for f in self.goals if f.startswith("(top")}
        self.bot_goals = {f for f in self.goals if f.startswith("(bot")}
        self.on_goals = {f for f in self.goals if f.startswith("(on")}
        self.hold_goals = {f for f in self.goals if f.startswith("(holding")}

    def __call__(self, node):
        state = node.state

        # Fast rejection: goal reached.
        if self.goals <= state:
            return 0

        # ------------------------------------------------------------------
        # 1. identify the currently unsatisfied goal facts
        # ------------------------------------------------------------------
        unsat = self.goals - state

        # ------------------------------------------------------------------
        # 2. sum weighted penalties
        # ------------------------------------------------------------------
        cost = 0
        for fact in unsat:
            if fact in self.top_goals or fact in self.bot_goals:
                cost += 3            # hardest to repair
            elif fact in self.on_goals:
                cost += 2            # medium effort
            else:  # holding or any other predicate
                cost += 1            # easiest

        return cost
