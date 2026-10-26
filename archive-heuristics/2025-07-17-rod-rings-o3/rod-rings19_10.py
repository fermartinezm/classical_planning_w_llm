from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def split_fact(f):
    """
    Helper:  '(top r1 s2)' -> ['top', 'r1', 's2']
    """
    return f[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    Greedy estimate for the Rod-Rings domain.  The domain allows the planner
    to *rotate* the sequence of rings on a stick by one position with every
    action (`put_top` or `put_bottom`).  An action therefore (a) inserts the
    currently held ring at **one** end of a chosen stick and (b) removes the
    opposite-end ring, which afterwards is held.  Intuitively, each action
    performs a single cyclic shift of the ring ordering of a stick.

    The heuristic reconstructs, for every stick, the current linear order
    of its rings (top … bottom).  For every goal fact it then estimates how
    many cyclic shifts are still necessary.  The estimate is cheap to
    compute yet considerably more informative than simply counting
    unsatisfied goal facts.

    Assumptions
    -----------
    • Every stick forms a single linear chain of rings
      (`top` fact + successive `on` facts until the `bot` ring).  
    • Exactly one ring can be held at a time (at most one `(holding …)` fact
      is true).  
    • A *rotation* (one heuristic cost unit) is sufficient to move a ring by
      one position either towards the top or the bottom of its stick.  
    • When a ring is already held, inserting it in the correct position is
      considered to cost one additional action.  Extracting a ring from a
      stick costs one action plus the rotations needed to bring it to an
      end of the stick.

    Heuristic Initialisation
    ------------------------
    During construction we
    1. store the already parsed goal facts,
    2. keep no static information – the domain declares none.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For every call on a search node we:
    1.  Parse the current state  
        – collect all `top`, `bot`, `on` and `holding` facts.
    2.  Rebuild, for every stick, a list `[r_top, …, r_bottom]`.  
        While doing so we memorise  
        • `location[ring] -> stick | 'hand'` and  
        • `pos[ring]      -> index` (0 = top).
    3.  For each *goal* fact we distinguish three cases:

        a)  (top R S) / (bot R S)  
            • already satisfied → 0  
            • R on S but not at the required end →  
              1 (final insertion) + min(cyclic distance)  
            • R held → 1 (insert)  
            • otherwise (R on different stick) → 3  
              (extract + insert, coarse estimate).

        b)  (on R₁ R₂)  
            • satisfied adjacency → 0  
            • R₁, R₂ on same stick but not adjacent → 1  
            • otherwise → 2.

    4.  Sum the individual estimates; the result is 0 only for goal states
        and finite for all reachable states.
    """

    def __init__(self, task):
        # Parse goal facts once.
        self.goal_facts = [split_fact(g) for g in task.goals]

    # --------------------------------------------------------------------- #
    #  Main heuristic                                                       #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # -------------------------- 1.  gather facts ---------------------- #
        top_of_stick = {}          # stick -> ring
        bot_of_stick = {}          # stick -> ring
        on_relation  = {}          # ring  -> ring (directly below)
        holding_ring = None

        for fact in state:
            pred, *args = split_fact(fact)
            if pred == "top":
                ring, stick = args
                top_of_stick[stick] = ring
            elif pred == "bot":
                ring, stick = args
                bot_of_stick[stick] = ring
            elif pred == "on":
                upper, lower = args
                on_relation[upper] = lower
            elif pred == "holding":
                holding_ring = args[0]

        # ------------------------ 2.  rebuild sequences ------------------- #
        stick_seq = {}             # stick -> [rings top … bottom]
        location  = {}             # ring  -> stick | 'hand'
        position  = {}             # ring  -> index inside stick list

        for stick, top_ring in top_of_stick.items():
            seq = [top_ring]
            current = top_ring
            while current in on_relation:
                current = on_relation[current]
                seq.append(current)
            stick_seq[stick] = seq
            for idx, r in enumerate(seq):
                location[r] = stick
                position[r] = idx

        if holding_ring:
            location[holding_ring] = "hand"
            position[holding_ring] = -1   # unused

        # ------------------------ 3.  evaluate goals ---------------------- #
        h = 0
        for pred, *args in self.goal_facts:
            # ------------------------------------------------------------- #
            # (top r s)  or  (bot r s)                                     #
            # ------------------------------------------------------------- #
            if pred in ("top", "bot"):
                ring, stick = args
                # Goal already satisfied?
                if (pred == "top" and top_of_stick.get(stick) == ring) or \
                   (pred == "bot" and bot_of_stick.get(stick) == ring):
                    continue

                # Ring currently held
                if location.get(ring) == "hand":
                    h += 1
                    continue

                # Ring already on the right stick
                if location.get(ring) == stick:
                    seq_len = len(stick_seq[stick])
                    idx     = position[ring]
                    target  = 0 if pred == "top" else seq_len - 1
                    # cyclic distance to target end
                    dist = abs(idx - target)
                    dist = min(dist, seq_len - dist)
                    h   += 1 + dist
                    continue

                # Ring sits on another stick – coarse estimate
                h += 3
            # ------------------------------------------------------------- #
            # (on r1 r2)                                                    #
            # ------------------------------------------------------------- #
            elif pred == "on":
                upper, lower = args
                if on_relation.get(upper) == lower:
                    continue  # already satisfied

                # Both rings on same stick?
                if location.get(upper) == location.get(lower) \
                   and location.get(upper) not in (None, "hand"):
                    h += 1
                else:
                    h += 2

            # ------------------------------------------------------------- #
            # Any other (not expected)                                      #
            # ------------------------------------------------------------- #
            else:
                # Fallback for unforeseen predicates – count them but keep finite.
                h += 1

        return h
