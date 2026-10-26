from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a PDDL fact into its atomic parts.  Example
        "(top r1 s2)"  --> ["top", "r1", "s2"]
    """
    return fact[1:-1].split()


def match(fact: str, *pattern):
    """
    Utility wrapper around ``fnmatch`` that allows us to write
    >>> match("(top r1 s2)", "top", "*", "s2")   #  True
    """
    return all(fnmatch(p, q) for p, q in zip(get_parts(fact), pattern))


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  The heuristic
    assigns an (inadmissible) cost to a state by counting how many goal
    predicates are still violated and estimating for every such predicate how
    many `put_top` / `put_bottom` actions are **at least** necessary to satisfy
    it.  The value is deliberately optimistic but distinguishes “easy-to-fix”
    violations (e.g. ring already in hand) from “harder” ones (ring buried
    somewhere on a different stick).

    Assumptions
    -----------
    1.  Exactly one ring is *always* held – an invariant of the domain.
    2.  A single `put_top` / `put_bottom` operation *inserts* the ring that is
        currently in hand and *extracts* the opposite end ring.  Therefore,
        *placing* a ring that is already held costs one action, while
        *acquiring* an arbitrary ring that sits on a stick costs (at least) one
        additional action.
    3.  We do **not** model interference between different goal facts; costs
        are summed independently.

    Heuristic Initialization
    ------------------------
    Only the set of goal facts has to be memorised.  There are no static facts
    in the *rod-rings* domain, therefore nothing else is stored.

    Step-By-Step Thinking for Computing `h(s)`
    -----------------------------------------
    1.  Extract from the current state
        • which ring is presently *held*,
        • for every stick its *top* ring and its *bottom* ring
          (needed to judge whether a ring is already on the correct stick).
    2.  Iterate over every goal fact `g`.
        • If `g` is already contained in the state, it contributes *0*.
        • Otherwise add a cost that approximates the **minimum** number of
          actions to achieve `g`:
          - `top r s` or `bot r s`
              1 action  if the held ring is `r`,
              2 actions if `r` is on stick `s` already,
              3 actions otherwise.
          - `on r1 r2`
              1 action  if the held ring is `r1`,
              2 actions otherwise.
          - `holding r`
              1 action  if `r` is the *top* or *bottom* ring of some stick,
              2 actions  otherwise.
          - Any other predicate (should not appear) → cost 1.
    3.  Return the sum of all individual costs.

    The heuristic is zero **iff** every goal predicate is satisfied and it
    returns a finite positive integer for all other (reachable) states.
    """

    def __init__(self, task):
        # Only the goal facts are required later on.
        self.goals = task.goals

    # --------------------------------------------------------------------- #
    #                               MAIN CALL                               #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ------------------------------------------------------------------
        # 1) Quick goal test – avoids any additional work
        # ------------------------------------------------------------------
        unsatisfied = self.goals - state
        if not unsatisfied:
            return 0

        # ------------------------------------------------------------------
        # 2) Extract convenience information from the state
        # ------------------------------------------------------------------
        held_ring = None
        top_of_stick = {}     # stick  -> ring
        bottom_of_stick = {}  # stick  -> ring

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "holding":
                held_ring = parts[1]

            elif pred == "top":
                ring, stick = parts[1], parts[2]
                top_of_stick[stick] = ring

            elif pred == "bot":
                ring, stick = parts[1], parts[2]
                bottom_of_stick[stick] = ring

        # Small helper -----------------------------------------------------
        def ring_on_same_stick(ring, stick):
            """Return *True* if we can prove that `ring` already sits on
               `stick` (by being its current top or bottom ring)."""
            return top_of_stick.get(stick) == ring or bottom_of_stick.get(stick) == ring

        # ------------------------------------------------------------------
        # 3) Estimate the cost for every unsatisfied goal fact
        # ------------------------------------------------------------------
        h = 0
        for fact in unsatisfied:
            pred, *args = get_parts(fact)

            if pred == "top":
                ring, stick = args
                if held_ring == ring:
                    h += 1
                elif ring_on_same_stick(ring, stick):
                    h += 2
                else:
                    h += 3

            elif pred == "bot":
                ring, stick = args
                if held_ring == ring:
                    h += 1
                elif ring_on_same_stick(ring, stick):
                    h += 2
                else:
                    h += 3

            elif pred == "on":
                upper, lower = args
                if held_ring == upper:
                    h += 1
                else:
                    h += 2  # optimistic lower bound

            elif pred == "holding":
                ring, = args
                if held_ring == ring:
                    h += 0   # would actually not be in `unsatisfied`
                elif ring in top_of_stick.values() or ring in bottom_of_stick.values():
                    h += 1
                else:
                    h += 2

            else:
                # Fallback cost for unforeseen predicates
                h += 1

        return h
