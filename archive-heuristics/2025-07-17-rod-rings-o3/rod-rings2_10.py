from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a fact like '(on r1 r2)' into ['on', 'r1', 'r2'].
    Parentheses are stripped and the remaining string is split on white-space.
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the **rod-rings** domain.  The heuristic
    tries to estimate how many `put_top` / `put_bottom` operations are still
    necessary to reach a goal state.  It does so by analysing how “deep” the
    involved rings are inside their current stacks and how far the stack
    configuration differs from the goal configuration.

    Assumptions
    -----------
    1. Every stick has exactly one *top* and one *bottom* ring at any time and
       the `on` relation forms a linked list from *top* to *bottom*.
    2. Each action can be abstractly regarded as
         – “remove a ring that is currently held” **and**
         – “insert this ring at one end of a stack”  
       while simultaneously  removing one of the current extreme rings and
       putting it in the hand.  Therefore every **non-trivial** modification of
       a stack costs at least one operator application.
    3. A ring that is already **held** is considered to be “free” (there are no
       rings above it) and can be placed somewhere with a single action.

    Heuristic Initialization
    ------------------------
    The constructor stores the set of goal facts for later comparison.  The
    domain contains no static predicates, so nothing else is required.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Reconstruct every current stack:
          • `top(r,s)` gives the first ring of stack *s*  
          • following the `on` chain leads downwards until no further `on`
            fact is found – the last ring encountered must equal `bot(r,s)`
       From this we compute for every ring  
       – `above_count`  : how many rings are above it in its stack  
       – `below_count`  : how many rings are below it in its stack  
       – `stack_of`     : on which stick the ring currently rests
    2.  Rings that occur only in a `holding` fact are treated as being in
        a special “hand” stack with depth 0 above and below.
    3.  For every goal fact that is **not yet satisfied** a local cost is
        estimated:
          • `top r s`   : `above_count[r] + 1`  
            (all rings above *r* must be removed, then one action to put *r*
            on top of stick *s*).
          • `bot r s`   : `below_count[r] + 1`  
            (all rings below *r* must be removed, then one action to put *r*
            at the bottom of stick *s*).
          • `on r1 r2`  : `above_count[r1] + above_count[r2] + 1`  
            (free both rings and perform one action to place *r1* on *r2*).
    4.  The heuristic value is the sum over all local costs.  
        Because every unsatisfied goal contributes **at least one** to the
        total, the heuristic is 0 **iff** the state already satisfies all goal
        conditions, is finite for every reachable state, and – while not
        admissible – provides informative guidance for a greedy best-first
        search.
    """

    def __init__(self, task):
        self.goals = task.goals  # type: frozenset[str]

    # --------------------------------------------------------------------- #
    #  Internals:  Analyse the current state                                #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _analyse_state(state):
        """
        Build helper structures that describe the configuration of every stick
        and the position of every ring inside its stack.
        Returns three dicts:
            above_count[ring]  – number of rings above `ring`
            below_count[ring]  – number of rings below `ring`
            stack_of[ring]     – the stick the ring is on (or None if held)
        """
        top_of = {}          # stick -> top ring
        next_down = {}       # ring  -> ring directly below it
        held_rings = set()   # rings currently in hand

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "top":       # (top r s)
                ring, stick = args
                top_of[stick] = ring
            elif pred == "on":      # (on r1 r2) : r1 on r2
                upper, lower = args
                next_down[upper] = lower
            elif pred == "holding":
                held_rings.add(args[0])

        above_count = {}
        below_count = {}
        stack_of = {}

        # Walk down every stack starting with its top ring
        for stick, first_ring in top_of.items():
            order = []
            current = first_ring
            visited = set()
            while current and current not in visited:
                visited.add(current)
                order.append(current)
                current = next_down.get(current)

            depth = len(order)
            for idx, ring in enumerate(order):
                above_count[ring] = idx
                below_count[ring] = depth - idx - 1
                stack_of[ring] = stick

        # Rings that are currently held are not part of any stack
        for ring in held_rings:
            above_count.setdefault(ring, 0)
            below_count.setdefault(ring, 0)
            stack_of.setdefault(ring, None)

        return above_count, below_count, stack_of

    # --------------------------------------------------------------------- #
    #  Heuristic evaluation                                                 #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # Fast exit: if the state already satisfies every goal, return 0
        if self.goals <= state:
            return 0

        # Build helper indices for the current state
        above, below, _ = self._analyse_state(state)

        h_value = 0
        for goal in self.goals:
            if goal in state:
                continue  # already achieved

            pred, *args = get_parts(goal)

            if pred == "top":           # (top r s)
                ring, _ = args
                cost = above.get(ring, 0) + 1

            elif pred == "bot":         # (bot r s)
                ring, _ = args
                cost = below.get(ring, 0) + 1

            elif pred == "on":          # (on r1 r2)
                r1, r2 = args
                cost = above.get(r1, 0) + above.get(r2, 0) + 1

            else:
                # Unknown predicate – fall back to a minimal positive cost
                cost = 1

            h_value += cost

        return h_value
