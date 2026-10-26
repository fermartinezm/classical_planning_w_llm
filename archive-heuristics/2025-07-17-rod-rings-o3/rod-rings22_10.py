from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Helper that removes the surrounding brackets from a PDDL fact and
    returns the individual components as a list of strings.

    Example
    -------
    >>> get_parts('(top r3 s2)')
    ['top', 'r3', 's2']
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  The heuristic
    returns an integer that (non-admissibly) estimates how many `put_top` /
    `put_bottom` actions are still required in order to reach a goal state.
    Each action always places the ring that is currently **held** either at the
    top or the bottom of some stick and, at the same time, removes the ring
    that occupied that position.  Consequently every operator is capable of
    satisfying at most two goal facts (e.g.  the correct *top* ring for a stick
    **and** the corresponding *on* relation).  The heuristic exploits this by
    counting, in a weighted fashion, how many goal facts are not yet true.

    Assumptions
    -----------
    1. Goal descriptions only contain predicates `top`, `bot` and `on`.
    2. Any additional facts – in particular the inevitable `(holding ?r)` fact –
       are allowed to remain true in the goal state (planner uses ⊆ test).
    3. Every solvable instance contains exactly one `(holding ?r)` fact in each
       reachable state (true for all benchmark instances).

    Heuristic Initialization
    ------------------------
    The constructor stores the *set* of goal facts because they never change
    during search.  No static facts are present in the domain, therefore no
    additional pre-processing is necessary.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given node / state the heuristic performs the following steps:

    1.  Identify the ring that is currently held (unique because of the domain
        semantics).

    2.  Partition the – still unsatisfied – goal facts into three groups:
        *top*, *bottom*, and *on* goals.

    3.  Assign a cost contribution to every unsatisfied goal fact
        • 2 for each missing *top* or *bottom* fact because one action is
          required to bring the desired ring into the gripper and a second
          action to place it at the correct end of its stick.
        • 1 for every unsatisfied direct adjacency *(on x y)* because such a
          relation can be established as a by-product when a ring is placed.

    4.  If the ring currently held happens to be **exactly** the ring that is
        needed for a still unsatisfied *top* or *bottom* fact, we subtract 1
        from the estimate – the first part of the work (getting the ring into
        the hand) is already done.

    5.  Return the resulting positive integer.  The value is `0` **iff** all
        goal facts are already satisfied, guaranteeing correctness of goal
        detection.
    """

    def __init__(self, task):
        # Store the (immutable) goal set for quick access during search.
        self.goals = task.goals

    # --------------------------------------------------------------------- #
    #  Main heuristic routine                                               #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # ------------------------------------------------------------
        #  1) find the ring that is currently held
        # ------------------------------------------------------------
        held_ring = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "holding":
                held_ring = parts[1]
                break

        # ------------------------------------------------------------
        #  2) classify unsatisfied goal facts
        # ------------------------------------------------------------
        unsatisfied_top = []      # (ring, stick)
        unsatisfied_bot = []      # (ring, stick)
        unsatisfied_on  = []      # (ring_above, ring_below)

        for goal in self.goals:
            if goal in state:        # already satisfied
                continue

            predicate, *args = get_parts(goal)
            if predicate == "top":
                unsatisfied_top.append(tuple(args))          # (ring, stick)
            elif predicate == "bot":
                unsatisfied_bot.append(tuple(args))
            elif predicate == "on":
                unsatisfied_on.append(tuple(args))           # (ring1, ring2)

        # ------------------------------------------------------------
        #  3) compute the raw cost estimate
        # ------------------------------------------------------------
        cost = (
            2 * (len(unsatisfied_top) + len(unsatisfied_bot))
            + 1 * len(unsatisfied_on)
        )

        # ------------------------------------------------------------
        #  4) useful ring already in hand?  -> small bonus
        # ------------------------------------------------------------
        if held_ring is not None:
            # If the held ring is *exactly* the one needed for a still missing
            # top/bottom fact we can skip the "pick-up" part of that effort.
            need_held_for_top = any(ring == held_ring for ring, _ in unsatisfied_top)
            need_held_for_bot = any(ring == held_ring for ring, _ in unsatisfied_bot)
            if need_held_for_top or need_held_for_bot:
                cost = max(0, cost - 1)   # keep heuristic non-negative

        return cost
