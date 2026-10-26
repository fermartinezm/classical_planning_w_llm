from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a PDDL fact into its components.

    Example
    -------
    "(top r1 s2)"  ->  ["top", "r1", "s2"]
    """
    return fact[1:-1].split()


class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the PDDL domain *rod-rings*.  It returns
    an estimate of how many **put_top / put_bottom** operations are still
    required to reach the goal.  The estimate is **not** admissible; it is
    tailored to guide a greedy best-first search while staying extremely cheap
    to compute.

    Assumptions
    -----------
    1. Every reachable state contains exactly one `(holding ?r)` fact
       (guaranteed by the two domain actions).
    2. Each stick has exactly one `top` ring and one `bot` ring; the rings
       between them are connected by a chain of `on` relations that is unique
       for that stick.
    3. Goal formulae only contain `top`, `bot`, and `on` predicates
       (as in the benchmark set).

    Heuristic Initialisation
    ------------------------
    The constructor analyses the goal facts once and partitions them into
    three convenient lists:
        • self.goal_tops :  [(ring, stick), …]
        • self.goal_bots :  [(ring, stick), …]
        • self.goal_ons  :  [(upper, lower), …]

    No static information exists in this domain, therefore nothing else has to
    be stored.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For every state `s` we perform the following steps.

    1.  Build two auxiliary data structures in **O(#rings)** time
        (a)  down_map[upper] = lower      for every  `(on upper lower)`
        (b)  ring_to_stick[ring] = stick  initialised from `top` and `bot`
        We then propagate stick identifiers through the `on` chains until the
        stick of every ring that is attached to *some* stick is known.

    2.  Score every goal literal:
        • Goal already satisfied →  0
        • `top r s`:
              – r is on stick s   →  +1      (only its height is wrong)
              – otherwise         →  +2
        • `bot r s`: analogous.
        • `on a b`:
              – a and b on same stick but not adjacent → +1
              – otherwise                              → +2
        The final heuristic value is the sum over all goal literals.

    Motivation of the costs
    -----------------------
    A single domain action can introduce at most **one** `top`, **one** `bot`
    and **one** `on` fact.  If a ring is already on the correct stick, only an
    additional rotation is necessary to reach the correct height (≈ one
    action ⇒ cost 1).  If it is on a different stick, at least two actions are
    needed (deposit elsewhere, pick up again ⇒ cost 2).  The same intuition
    is applied to adjacency (`on`) relations.

    Properties
    ----------
    • Returns 0 on every goal state.
    • Returns a finite integer on every reachable state.
    • Very low computational overhead (linear in the number of rings).
    """

    # --------------------------------------------------------------------- #
    #  Constructor                                                          #
    # --------------------------------------------------------------------- #
    def __init__(self, task):
        self.goal_tops = []
        self.goal_bots = []
        self.goal_ons = []

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                ring, stick = args
                self.goal_tops.append((ring, stick))
            elif pred == "bot":
                ring, stick = args
                self.goal_bots.append((ring, stick))
            elif pred == "on":
                upper, lower = args
                self.goal_ons.append((upper, lower))

    # --------------------------------------------------------------------- #
    #  Helper: reconstruct stick mapping                                    #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _compute_ring_to_stick(state):
        """Return two dicts:  ring → stick   and   upper → lower (down_map)."""
        ring_to_stick = {}
        down_map = {}

        # 1) direct information
        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "top" or pred == "bot":
                ring, stick = args
                ring_to_stick[ring] = stick
            elif pred == "on":
                upper, lower = args
                down_map[upper] = lower

        # 2) propagate stick IDs along the on-chain
        changed = True
        while changed:
            changed = False
            for upper, lower in down_map.items():
                if upper in ring_to_stick and lower not in ring_to_stick:
                    ring_to_stick[lower] = ring_to_stick[upper]
                    changed = True
                elif lower in ring_to_stick and upper not in ring_to_stick:
                    ring_to_stick[upper] = ring_to_stick[lower]
                    changed = True

        return ring_to_stick, down_map

    # --------------------------------------------------------------------- #
    #  The heuristic itself                                                 #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # Early exit for goal states (cheap check, many planners do it first)
        if node.task.goal_reached(state):
            return 0

        ring_to_stick, down_map = self._compute_ring_to_stick(state)

        cost = 0

        # 1) TOP goals
        for ring, stick in self.goal_tops:
            if f"(top {ring} {stick})" in state:
                continue
            elif ring_to_stick.get(ring) == stick:
                cost += 1
            else:
                cost += 2

        # 2) BOT goals
        for ring, stick in self.goal_bots:
            if f"(bot {ring} {stick})" in state:
                continue
            elif ring_to_stick.get(ring) == stick:
                cost += 1
            else:
                cost += 2

        # 3) ON goals
        for upper, lower in self.goal_ons:
            if f"(on {upper} {lower})" in state:
                continue

            same_stick = (
                ring_to_stick.get(upper) is not None
                and ring_to_stick.get(upper) == ring_to_stick.get(lower)
            )

            if same_stick:
                cost += 1
            else:
                cost += 2

        return cost
