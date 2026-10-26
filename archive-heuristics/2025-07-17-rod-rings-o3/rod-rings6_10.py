from heuristics.heuristic_base import Heuristic


def strip_fact(fact: str):
    """
    Helper that removes the surrounding parentheses of a fact and
    returns a tuple of its components, e.g.

        "(top r1 s2)"  -> ("top", "r1", "s2")
    """
    return tuple(fact[1:-1].split())


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  The heuristic
    returns an (inadmissible) estimate of how many `put_top` / `put_bottom`
    actions are still necessary to reach a goal state.  The estimate is
    intentionally simple but tailored to the structure of the domain: every
    ring that is not yet in its required final relationship (‛top’, ‛bot’ or
    ‛on’) will have to be moved at least once, therefore the heuristic counts
    how many such rings remain.

    Assumptions
    -----------
    • Only the predicates `top`, `bot` and `on` appear in goal descriptions  
    • Each *put*-action can place **exactly one** ring in its final position  
    • No action can fix two rings that are both currently misplaced in one
      step – at least one separate action is required for each of them.  
    • A ring that is already in all of its goal relationships will never have
      to be moved again.

    Heuristic Initialization
    ------------------------
    During construction the heuristic
    1. stores the complete set of goal facts,
    2. groups them by the ring they refer to (fast lookup at evaluation time).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node the heuristic is computed as follows:

    1.  Build a `set` with all facts that are true in the current state
        (this is provided to us as `node.state`).

    2.  Iterate over all rings that are mentioned in the goal description.
        For each ring:
        • collect the *goal* facts that reference this ring;  
        • check whether **all** of them are contained in the current state.

    3.  If even one goal fact for this ring is still missing the ring is
        considered *misplaced* and contributes the constant cost 1 to the
        heuristic value (because at least one more `put_*` action will be
        necessary to move it into its final place).

    4.  The resulting value is the number of misplaced rings.  
        A goal state satisfies all goal facts ⇒ 0 misplaced rings ⇒ heuristic
        value 0 – conforming to the required property `h(s_goal) = 0`.

    Even though the heuristic is a crude under-approximation of the true
    distance, its domain-specific aggregation on a *per-ring* basis provides
    more guidance than a plain “unsatisfied-fact counter” while still being
    computable in O(|goals|) time.
    """

    def __init__(self, task):
        # Store the goal facts.
        self.goals = frozenset(task.goals)

        # Pre-group goal facts by the ring they refer to
        self.goal_facts_per_ring = {}
        for fact in self.goals:
            pred, *args = strip_fact(fact)
            if pred in {"top", "bot"}:
                ring = args[0]           # (top r s) or (bot r s)
            elif pred == "on":
                ring = args[0]           # (on r_above r_below) – we count the
                                          #        ring that has to be placed
            else:                        # Any other predicate – should not
                continue                 # appear in goals for this domain
            self.goal_facts_per_ring.setdefault(ring, set()).add(fact)

    def __call__(self, node):
        state = node.state

        misplaced_rings = 0

        for ring, required_facts in self.goal_facts_per_ring.items():
            if not required_facts <= state:          # at least one missing fact
                misplaced_rings += 1

        return misplaced_rings
