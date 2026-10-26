from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Helper that strips the surrounding parentheses of a PDDL fact
    and returns the blank–separated list of components.
    Example: "(top r2 s1)"  -->  ["top", "r2", "s1"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–specific heuristic for the *rod-rings* domain.  Every stick
    always contains exactly two rings – one marked *top* and one marked
    *bottom* – and exactly one further ring is in the robot’s hand
    (`holding`).  The two operators `put_top` and `put_bottom` exchange
    the ring in hand with the *top* or *bottom* ring of any stick,
    thereby performing a 3-cycle (hand → top → bottom → hand).  In a
    single step we can therefore (i) establish the desired top ring or
    (ii) establish the desired bottom ring of one stick, but not both
    simultaneously.  The heuristic estimates how many such 3-cycle
    exchanges are still required.

    Assumptions
    -----------
    1. Each stick contains **exactly one top and one bottom** ring in
       every reachable state (this is enforced by the domain
       invariants).
    2. Exactly **one** ring is in the robot’s hand at any time.
    3. A single operator invocation can satisfy at most **one** of the
       two position predicates (top / bottom) of a stick.  This allows
       us to approximate the remaining distance by counting, per stick,
       how many of those predicates are still unsatisfied.
    4. The heuristic is *inadmissible* on purpose – it trades absolute
       correctness for speed and guidance.

    Heuristic Initialization
    ------------------------
    During construction we pre-parse the goal description once and store
    it in four dedicated structures:

      • `goal_top[stick]  ->  ring`   – the required top ring of *stick*
      • `goal_bot[stick]  ->  ring`   – the required bottom ring
      • `goal_on[upper]   ->  lower`  – all explicit on-relations that do
                                        are **not** implied by the top /
                                        bottom information
      • `goal_holding`               – the ring that must be held (if any)

    This pre-processing avoids heavy string matching inside the hot
    heuristic loop.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Scan the current state once and extract
        • the *currently* held ring,
        • the *current* top and bottom ring of every stick, and
        • the explicit `on` relations.
    2.  For every stick that is mentioned in the goal:
        a) Test whether the current top ring already equals the required
           top ring (if any).
        b) Test whether the current bottom ring already equals the
           required bottom ring (if any).
        c) 0 of the two predicates wrong → cost 0  
           1 of the two predicates wrong → cost 1  
           2 of the two predicates wrong → cost 2
    3.  If the goal demands a specific ring to be *held*, add **1** more
        action if that is currently not the case.
    4.  For every explicit `on upper lower` goal that is **not yet true**
        add **1** additional action.  (These pairs are only counted when
        the goal did not already specify the corresponding top / bottom
        predicate – hence no double counting takes place.)
    5.  The sum of the contributions from (2)–(4) is returned as the
        heuristic value.  It is 0 in any goal state, strictly positive
        otherwise, and always finite for reachable states.
    """

    def __init__(self, task):
        self.goals = task.goals

        # ---- Parse goal description once ---------------------------------
        self.goal_top = {}          # stick -> ring
        self.goal_bot = {}          # stick -> ring
        self.goal_on = {}           # upper_ring -> lower_ring
        self.goal_holding = None    # ring

        for fact in self.goals:
            pred, *args = get_parts(fact)

            if pred == "top":
                ring, stick = args
                self.goal_top[stick] = ring

            elif pred == "bot":
                ring, stick = args
                self.goal_bot[stick] = ring

            elif pred == "holding":
                (ring,) = args
                self.goal_holding = ring

            elif pred == "on":
                upper, lower = args
                self.goal_on[upper] = lower

        # Remove on-relations that are *implicitly* covered by an explicit
        # top/bot goal.  (When the goal says “top r s” we also know that
        # “on r bot(s)” must hold – counting both would over-penalise.)
        implicit_ons = set()
        for stick, ring_top in self.goal_top.items():
            ring_bottom = self.goal_bot.get(stick)
            if ring_bottom:                        # only if bottom given
                implicit_ons.add((ring_top, ring_bottom))

        # Delete those implicit pairs from goal_on
        for upper, lower in list(self.goal_on.items()):
            if (upper, lower) in implicit_ons:
                del self.goal_on[upper]

    # ---------------------------------------------------------------------

    def __call__(self, node):
        state = node.state

        # ---- Current state information -----------------------------------
        current_top = {}       # stick -> ring
        current_bot = {}       # stick -> ring
        current_on = {}        # upper_ring -> lower_ring
        current_holding = None

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "top":
                ring, stick = args
                current_top[stick] = ring

            elif pred == "bot":
                ring, stick = args
                current_bot[stick] = ring

            elif pred == "holding":
                (ring,) = args
                current_holding = ring

            elif pred == "on":
                upper, lower = args
                current_on[upper] = lower

        # ---- Heuristic computation ---------------------------------------
        h = 0

        # 1. Stick-wise contribution
        sticks_of_interest = set(self.goal_top.keys()) | set(self.goal_bot.keys())
        for stick in sticks_of_interest:
            top_ok = (stick not in self.goal_top) or \
                     (current_top.get(stick) == self.goal_top[stick])

            bot_ok = (stick not in self.goal_bot) or \
                     (current_bot.get(stick) == self.goal_bot[stick])

            unsat = (not top_ok) + (not bot_ok)   # 0, 1 or 2
            h += unsat                           # cost as explained above

        # 2. Holding contribution
        if self.goal_holding is not None and current_holding != self.goal_holding:
            h += 1

        # 3. Remaining explicit on-relations
        for upper, lower in self.goal_on.items():
            if current_on.get(upper) != lower:
                h += 1

        return h