from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Helper: remove the outer brackets of a PDDL fact and split the remaining
    string into its individual symbols.
    Example
        "(top r1 s2)"     -> ["top", "r1", "s2"]
        "(on  r3  r4)"    -> ["on", "r3", "r4"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  It assigns an
    (inadmissible) estimate of how many `put_top` / `put_bottom` actions are
    still required to satisfy every goal literal.

    Assumptions
    -----------
    1. Every state contains **exactly one** `(holding ?r)` fact (this is
       enforced by the domain design).
    2. One action can *usually* establish at most one missing goal predicate
       (sometimes two, but never more).  In particular …
          • a single action can achieve an `on` relation **only** if the upper
            ring is already in the hand **and** the lower ring is the current
            top of some stick.
          • a single action can put a held ring to the top (or bottom)
            position of a stick, thereby achieving a `top` or `bot` goal.
       If those “lucky” pre-conditions are not met, at least two actions will
       be necessary.

    Heuristic Initialization
    ------------------------
    The constructor stores the set of goal literals so that the heuristic
    itself only has to look at the current state while being evaluated.  No
    static information is present in this domain, thus no further
    preprocessing is required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search `node`, the heuristic works as follows:

    1.  Build quick lookup tables that describe the *current* position of every
        ring:

            •  `held_ring`              – the ring that is held
            •  `tops[ring]   = stick`   – `ring` is the top of `stick`
            •  `bottoms[ring] = stick`  – `ring` is the bottom of `stick`

        All other rings are somewhere in the middle of a pile.

    2.  Iterate over all goal literals.  If a literal is already satisfied, it
        contributes nothing.  Otherwise:
            •  Unsatisfied `top` / `bot`
                   –  If the ring is in the hand            → +1
                   –  Otherwise                            → +2
            •  Unsatisfied `on`
                   –  If   (upper ring is in hand **and**
                         lower ring is a current top)      → +1
                   –  Otherwise                            → +2

        The cost table is a coarse but inexpensive approximation of the number
        of actions still needed to establish the literal.

    3.  Sum all individual costs – this is the heuristic value `h(s)`.  The
        value is zero *iff* every goal literal already holds, and it is always
        finite for solvable states.

    The heuristic is designed to be dirt-cheap to compute (linear in the
    number of facts) while still giving search a meaningful preference for
    states that “line things up” for an inexpensive single-action fix.
    """

    def __init__(self, task):
        self.goals = task.goals  # Store for quick access

    # ---------------------------------------------------------------------- #
    #  Main heuristic                                                        #
    # ---------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ------------------------------------------------------------------ #
        #  1.  Analyse current state                                         #
        # ------------------------------------------------------------------ #
        held_ring = None
        tops, bottoms = {}, {}

        for fact in state:
            predicate, *args = get_parts(fact)

            if predicate == "holding":
                held_ring, = args

            elif predicate == "top":
                ring, stick = args
                tops[ring] = stick

            elif predicate == "bot":
                ring, stick = args
                bottoms[ring] = stick

            # `on` relations are only needed for goal checking (done later)

        # ------------------------------------------------------------------ #
        #  2.  Count the remaining work                                      #
        # ------------------------------------------------------------------ #
        cost = 0
        unsatisfied = self.goals - state

        for fact in unsatisfied:
            predicate, *args = get_parts(fact)

            # -------------------  TOP / BOT goals  ------------------------ #
            if predicate in ("top", "bot"):
                ring, _stick = args

                if ring == held_ring:
                    cost += 1          # One placement action will do
                else:
                    cost += 2          # Need to get ring into hand first

            # -------------------------  ON goals  ------------------------- #
            elif predicate == "on":
                upper, lower = args

                # Fast “lucky” case: upper ring is in hand *and* lower is top
                if upper == held_ring and lower in tops:
                    cost += 1
                else:
                    cost += 2

            # Fallback (should not be needed, but keep the heuristic robust)
            else:
                cost += 1

        return cost
