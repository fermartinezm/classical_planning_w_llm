from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Split a fact into its components while stripping the surrounding
    parentheses, e.g. '(top r1 s2)'  ->  ['top', 'r1', 's2'].
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A simple, efficiently–computable domain–dependent heuristic for the
    `rod-rings` domain.  The heuristic counts how many TOP, BOT and ON goal
    facts are still unsatisfied and assigns each of them a small weight that
    roughly reflects the minimum number of actions that will be required to
    achieve the outstanding facts.

    Assumptions
    -----------
    •  Each action (`put_top`, `put_bottom`) can satisfy at most one TOP/BOT
       fact and at most one ON fact.
    •  There is at most one ring in the robot’s hand and the goal never
       requires a particular ring to be **held** (the benchmark problems used
       so far confirm this, but the code still supports a *holding* goal).

    Heuristic Initialization
    ------------------------
    In the constructor the goal set is partitioned into three easy-to-handle
    categories:
        – TOP goals   :  (top  <ring> <stick>)
        – BOT goals   :  (bot  <ring> <stick>)
        – ON  goals   :  (on   <ring_above> <ring_below>)
        – HOLDING     :  (holding <ring>)      (optional)
    No static facts are defined in the domain, therefore no further
    preprocessing is necessary.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  If all goal facts are already present, return 0.
    2.  Extract the current state information into three tiny lookup sets
        mirroring the goal partition (tops, bots, ons) and a variable for the
        ring that is currently held (if any).
    3.  Count how many TOP, BOT, ON and (optional) HOLDING goals are missing.
    4.  Multiply each counter by a small weight that reflects how many actions
        are *at least* necessary:
            – TOP   and BOT  goals need ≥ 1 action each  →  weight 2
            – ON    goals may be achieved “for free” when satisfying
              TOP/BOT but sometimes need an extra move   →  weight 1
            – HOLDING goals (rare)                        weight 1
    5.  Return the weighted sum as heuristic value.

    The heuristic is deliberately inadmissible (weights > 1 for TOP/BOT) but
    inexpensive to compute and informed enough to reduce the number of node
    expansions in greedy best-first search.
    """

    # ---------------------  construction --------------------------------- #

    def __init__(self, task):
        # Partition the goal set for fast look-ups during search.
        self.goal_tops = set()
        self.goal_bots = set()
        self.goal_ons = set()
        self.goal_holding = None

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                self.goal_tops.add(tuple(args))       # (ring, stick)
            elif pred == "bot":
                self.goal_bots.add(tuple(args))       # (ring, stick)
            elif pred == "on":
                self.goal_ons.add(tuple(args))        # (ring_above, ring_below)
            elif pred == "holding":
                # There is at most one such goal.  Store the ring name.
                self.goal_holding = args[0]

        # Cache the whole goal set for a fast goal test.
        self.goal_set = task.goals

    # ---------------------  heuristic evaluation  ------------------------ #

    def __call__(self, node):
        state = node.state

        # Fast goal test
        if self.goal_set <= state:
            return 0

        # --- extract current TOP / BOT / ON / HOLDING information ----------
        current_tops = set()
        current_bots = set()
        current_ons = set()
        current_holding = None

        for fact in state:
            pred, *args = get_parts(fact)
            if pred == "top":
                current_tops.add(tuple(args))
            elif pred == "bot":
                current_bots.add(tuple(args))
            elif pred == "on":
                current_ons.add(tuple(args))
            elif pred == "holding":
                current_holding = args[0]

        # --- count unsatisfied goal facts ----------------------------------
        missing_tops = len(self.goal_tops - current_tops)
        missing_bots = len(self.goal_bots - current_bots)
        missing_ons = len(self.goal_ons - current_ons)

        missing_holding = 0
        if self.goal_holding is not None and self.goal_holding != current_holding:
            missing_holding = 1

        # --- weighted sum ---------------------------------------------------
        # Weights chosen empirically:
        #   TOP/BOT goals  ->  2
        #   ON goals       ->  1
        #   HOLDING goal   ->  1
        return 2 * (missing_tops + missing_bots) + missing_ons + missing_holding
