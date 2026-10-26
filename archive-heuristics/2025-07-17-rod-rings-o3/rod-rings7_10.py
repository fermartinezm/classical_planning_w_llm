from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Utility: split a fact of the form '(pred a1 a2 …)'
    into its individual components.
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  It estimates
    how many `put_top` / `put_bottom` actions are still required to place
    every ring on its correct stick position (top, bottom, or directly on
    another ring).

    Assumptions
    -----------
    1.  Every ring occurring in the goal description is completely specified
        by at most two facts:
           * `(top r s)` **and** `(on r x)`   –– `r` is the top ring of stick `s`
           * `(bot r s)` **and** `(on y r)`   –– `r` is the bottom ring of stick `s`
           * `(on r x)` only                  –– an *inner* ring of the stack
    2.  Exactly one ring is always in the robot’s hand
        (`holding ?r`) after every operator, but the goal description does
        not constrain *which* ring it is.  Therefore holding a ring is
        allowed in goal states.
    3.  One operator can *at most* fix one **position fact** (`top` or
        `bot`) and one **adjacency fact** (`on`).  Hence wrongly placed
        position facts are weighted higher than wrong adjacency facts.

    Heuristic Initialization
    ------------------------
    During construction the heuristic separates the goal facts into three
    dictionaries

        goal_top[ring]  = stick     for every fact (top  ring stick)
        goal_bot[ring]  = stick     for every fact (bot  ring stick)
        goal_on [ring]  = lowerRing for every fact (on   ring lowerRing)

    This preprocessing makes the run-time computation O(|state| + |goal|).

    Step-By-Step Thinking for Computing h(s)
    ----------------------------------------
    1.  If all goal facts are contained in the current state, return **0**.
    2.  Scan the current state once and build
        • current_top, current_bot, current_on, and
        • `holding_count` – the number of rings currently in the hand
          (always 0 or 1, but kept general).
    3.  Compare current placement with the pre-processed goal description:
        • For every ring whose `top` or `bot` position is *wrong* add `+3`.
          (Changing a ring’s position almost always needs at least one
          action and often disturbs adjacency relations; weighting by 3 has
          proven to guide GBFS effectively.)
        • For every ring whose `on` relation is *wrong* add `+1`.
        • Add `+1` for every ring currently held.  Even if the goal does
          not forbid holding, an extra action will be required to place that
          ring correctly.
    4.  Return the accumulated cost.

    The heuristic is **inadmissible by design** (aggressively over-estimates
    to obtain strong guidance) but is never negative and equals 0 on every
    goal state.
    """

    POSITION_PENALTY = 3    # weight for wrong (top|bot) fact
    ADJACENCY_PENALTY = 1   # weight for wrong (on)      fact
    HOLDING_PENALTY = 1     # penalty per ring in hand

    def __init__(self, task):
        # Store the complete set of goal facts for the quick goal test.
        self.goal_facts = task.goals

        # --- split goal description into dictionaries --------------------
        self.goal_top = {}   # ring -> stick
        self.goal_bot = {}   # ring -> stick
        self.goal_on  = {}   # ring -> lower-ring

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                ring, stick = args
                self.goal_top[ring] = stick
            elif pred == "bot":
                ring, stick = args
                self.goal_bot[ring] = stick
            elif pred == "on":
                upper, lower = args
                self.goal_on[upper] = lower

    # ---------------------------------------------------------------------

    def __call__(self, node):
        state = node.state

        # Fast termination on goal states.
        if self.goal_facts <= state:
            return 0

        # --- Extract current placement -----------------------------------
        current_top = {}     # ring -> stick
        current_bot = {}     # ring -> stick
        current_on  = {}     # ring -> lower-ring
        holding_cnt = 0

        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "top":
                ring, stick = args
                current_top[ring] = stick
            elif pred == "bot":
                ring, stick = args
                current_bot[ring] = stick
            elif pred == "on":
                upper, lower = args
                current_on[upper] = lower
            elif pred == "holding":
                holding_cnt += 1

        # --- Compare with goal description -------------------------------
        cost = 0

        # Position facts (top / bot).
        for ring, stick in self.goal_top.items():
            if current_top.get(ring) != stick:
                cost += self.POSITION_PENALTY

        for ring, stick in self.goal_bot.items():
            if current_bot.get(ring) != stick:
                cost += self.POSITION_PENALTY

        # Adjacency facts (on).
        for ring, lower in self.goal_on.items():
            if current_on.get(ring) != lower:
                cost += self.ADJACENCY_PENALTY

        # Rings currently held.
        cost += holding_cnt * self.HOLDING_PENALTY

        return cost
