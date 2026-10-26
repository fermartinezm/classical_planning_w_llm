from heuristics.heuristic_base import Heuristic


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent, *inadmissible but informative* heuristic for the
    `rod-rings` domain.  The value returned is an estimate of the number
    of `put_top` / `put_bottom` actions that are still required to reach
    the goal.

    Assumptions
    -----------
    1. Exactly one fact of the form `(holding ?r)` is true in every
       reachable state – the agent always carries **one** ring.
    2. Every stick contains precisely two rings: one at the top and one
       at the bottom, linked by a single `(on top bottom)` predicate.
    3. One `put_*` operator
         • places the carried ring at one end of a stick **and**
         • removes the ring that was at the opposite end, putting it in
           the agent’s hand.  
       Hence a single action can achieve up to two goal literals.

    Heuristic Initialization
    ------------------------
    Only the set of goal literals is required at run-time.  Therefore
    the constructor merely stores `task.goals`.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  If **all** goal literals are already satisfied, the heuristic
        value is **0**.
    2.  Determine the ring that is currently being held
        (the one occurring in the single `(holding ?r)` fact).
    3.  For each goal literal that is *not* yet satisfied:
        • `(top R S)` or `(bot R S)`
            - If the agent is *already* holding `R`, one action suffices
              → contribution = 1.
            - Otherwise at least one swap is needed to pick up `R`
              *plus* one action to place it  
              → contribution = 2.
        • `(on R1 R2)`
            - Although most `on` relations will be established together
              with a `top`/`bot` placement, we pessimistically assign
              one additional action  
              → contribution = 1.
        • Any other (unexpected) predicate is counted as one action.
    4.  The heuristic value is the sum of all individual contributions.
        It is finite for every solvable state and equals `0` in goal
        states.

    The computation is purely syntactic (`O(|goals|)`) and therefore
    negligible compared to successor generation.
    """

    def __init__(self, task):
        self.goals = task.goals  # store goal literals

    @staticmethod
    def _parts(fact: str):
        """Return ['pred', 'arg1', ...] for a fact '(pred arg1 …)'."""
        return fact[1:-1].split()

    def __call__(self, node):
        state = node.state

        # Early exit: goal already reached
        if self.goals <= state:
            return 0

        # Identify the ring that is currently held
        held_ring = None
        for f in state:
            if f.startswith("(holding "):
                held_ring = self._parts(f)[1]
                break

        cost = 0
        for goal in self.goals:
            if goal in state:        # already satisfied
                continue

            pred, *args = self._parts(goal)

            if pred in ("top", "bot"):
                ring = args[0]
                cost += 1 if ring == held_ring else 2

            elif pred == "on":
                cost += 1            # pessimistic single-action estimate

            else:
                cost += 1            # fallback for unforeseen predicates

        return cost
