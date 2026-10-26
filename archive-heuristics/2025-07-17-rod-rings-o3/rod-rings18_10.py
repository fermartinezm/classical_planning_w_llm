from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


# ---------------------------------------------------------------------------
# Helper functions that work on the PDDL facts (strings).
# ---------------------------------------------------------------------------
def get_parts(fact):
    """
    Return the list of symbols that make up a fact.

    Example
    -------
    "(top r1 s2)"  -->  ["top", "r1", "s2"]
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Wild-card matching on facts.  Every element of *pattern is compared to the
    corresponding element of the fact.  The pattern may use '*' as a wildcard.
    """
    return all(fnmatch(p, pat) for p, pat in zip(get_parts(fact), pattern))


# ---------------------------------------------------------------------------
# Domain–dependent heuristic
# ---------------------------------------------------------------------------
class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent, *non-admissible* heuristic for the               │
    “rod-rings’’ domain.  The heuristic assigns an estimated number of   │
    actions that are still necessary to reach the goal by inspecting     │
    every individual goal literal and judging how “far’’ it is from the  │
    current state.

    Assumptions
    -----------
    •  Only one ring can be held at a time (as enforced by the domain).  
    •  Every `put_top` / `put_bottom` action exchanges the currently held
       ring with one of the two end-rings of a stick, thereby satisfying
       at most *one* goal literal.                                       
    •  The cost contribution of different goal literals is assumed to be
       additive – interactions between them are ignored.  Consequently
       the heuristic is *inadmissible* but still provides good
       guidance for Greedy Best-First Search.

    Heuristic Initialization
    ------------------------
    The constructor pre-processes the task’s goal set and stores four
    dedicated lists that contain:
        – desired (top ring stick) facts
        – desired (bot ring stick) facts
        – desired (on ring₁ ring₂) facts
        – desired (holding ring) facts
    All other information is dynamic and therefore analysed inside the
    `__call__` method.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1.  From the state build quick-access data structures:
        • the ring that is currently held (if any);
        • a mapping   top_ring → stick  for all sticks;
        • a mapping bottom_ring → stick  for all sticks;
        • the set of current (on x y) pairs.
    2.  For every goal literal that is **already** satisfied add **0** to
        the heuristic.
    3.  Otherwise estimate how many actions are still required:

        Goal (top r s)
            0  if already satisfied
            1  if r is currently *held*                      (one put_top)
            2  if r is on stick s but not at the top
            3  otherwise

        Goal (bot r s)
            0  if already satisfied
            1  if r is currently *held*                    (one put_bottom)
            2  if r is on stick s but not at the bottom
            3  otherwise

        Goal (on r₁ r₂)
            0  if already satisfied
            1  if r₁ is *held* **and** r₂ is the current top of a stick
            2  if (r₁ is held) xor (r₂ is current top)
            3  otherwise

        Goal (holding r)
            0  if already satisfied
            1  otherwise        (we must pick the ring up once)

    4.  The final heuristic value is the sum over the individual
        contributions – this value is **0** only in goal states and is
        always finite for solvable states.
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(self, task):
        self.goals = task.goals

        # Partition the goals according to their predicate for fast access.
        self.goal_top      = []     # [(ring, stick), …]
        self.goal_bot      = []     # [(ring, stick), …]
        self.goal_on       = []     # [(ring1, ring2), …]
        self.goal_holding  = set()  # {ring, …}

        for fact in self.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                self.goal_top.append(tuple(args))          # (ring, stick)
            elif pred == "bot":
                self.goal_bot.append(tuple(args))          # (ring, stick)
            elif pred == "on":
                self.goal_on.append(tuple(args))           # (ring1, ring2)
            elif pred == "holding":
                self.goal_holding.add(args[0])

    # ---------------------------------------------------------------------
    # Heuristic evaluation
    # ---------------------------------------------------------------------
    def __call__(self, node):
        state = node.state
        cost  = 0

        # --- Extract dynamic information from the state -----------------
        held_ring = None
        top_of    = {}      # ring -> stick
        bottom_of = {}      # ring -> stick
        on_pairs  = set()   # {(ringA, ringB), …}

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "holding":
                held_ring = args[0]

            elif pred == "top":
                ring, stick = args
                top_of[ring] = stick

            elif pred == "bot":
                ring, stick = args
                bottom_of[ring] = stick

            elif pred == "on":
                ring1, ring2 = args
                on_pairs.add((ring1, ring2))

        # --- Evaluate TOP goals -----------------------------------------
        for ring, stick in self.goal_top:
            if match_in_state(state, "top", ring, stick):
                continue                                   # already true
            if held_ring == ring:
                cost += 1                                  # one put_top
            elif bottom_of.get(ring) == stick or any(on_pairs_match(ring, stick, on_pairs, top_of)):
                cost += 2
            else:
                cost += 3

        # --- Evaluate BOT goals -----------------------------------------
        for ring, stick in self.goal_bot:
            if match_in_state(state, "bot", ring, stick):
                continue
            if held_ring == ring:
                cost += 1                                  # one put_bottom
            elif top_of.get(ring) == stick or any(on_pairs_match_bottom(ring, stick, on_pairs, bottom_of)):
                cost += 2
            else:
                cost += 3

        # --- Evaluate ON goals ------------------------------------------
        for r1, r2 in self.goal_on:
            if (r1, r2) in on_pairs:
                continue
            r2_is_top = r2 in top_of
            if held_ring == r1 and r2_is_top:
                cost += 1
            elif held_ring == r1 or r2_is_top:
                cost += 2
            else:
                cost += 3

        # --- Evaluate HOLDING goals -------------------------------------
        for ring in self.goal_holding:
            if held_ring == ring:
                continue
            cost += 1

        return cost


# ---------------------------------------------------------------------------
# Small utility helpers used inside the heuristic
# ---------------------------------------------------------------------------
def match_in_state(state, predicate, *args):
    """
    Convenience wrapper that checks whether the given fact is contained in the
    state.  Wild-cards are **not** allowed; the fact has to match exactly.
    """
    return f"({predicate} {' '.join(args)})" in state


def on_pairs_match(ring, stick, on_pairs, top_of):
    """
    Generator that yields `True` iff the ring is somewhere on the *same* stick.
    Used to decide whether the ring is “close’’ to the desired position.
    """
    for (r_above, r_below) in on_pairs:
        if r_above == ring and top_of.get(r_below) == stick:
            yield True


def on_pairs_match_bottom(ring, stick, on_pairs, bottom_of):
    """
    Similar helper for detecting whether the ring is somewhere on stick `stick`
    but not yet at the bottom.
    """
    for (r_above, r_below) in on_pairs:
        if r_below == ring and bottom_of.get(r_above) == stick:
            yield True
