from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Utility that removes the surrounding brackets of a PDDL
    fact and splits it into its atomic components.

    Example
    -------
    "(top r1 s2)"   ->  ["top", "r1", "s2"]
    "(on r2 r4)"    ->  ["on",  "r2", "r4"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  It estimates
    the number of `put_top` / `put_bottom` actions that are still required
    to reach the goal state.

    Assumptions
    -----------
    1. Every action inserts the ring that is currently held at one end of a
       stick and removes the opposite end ring, so **one action can create at
       most two goal literals**:
        • either a  (top  R S)  together with  (on  R X)      or
        • a        (bot  R S)  together with  (on  X R)
       where *R* is the just-inserted ring and *X* the displaced one.
    2. Satisfying a `(top …)` and its accompanying `(on …)` (or respectively
       `(bot …)` and `(on …)`) with a *single* operator is the cheapest way
       to achieve both, therefore the heuristic pairs such literals whenever
       possible.
    3. No static facts exist in this domain, so all information that matters
       is contained in the mutable part of the state.

    Heuristic Initialization
    ------------------------
    While building the heuristic object we parse the *goal facts* once and
    store them in three dedicated containers:
        • self.top_goals : {ring → stick}
        • self.bot_goals : {ring → stick}
        • self.on_goals  : [(upper_ring , lower_ring)]
    This preprocessing allows the actual heuristic computation to run in
    time O(#goal-facts).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search node the heuristic proceeds as follows

    1.  Determine which goal facts are already satisfied by simple set
        membership tests (`fact in state`).  All others form three *unsatisfied*
        collections:
            –   unsat_top : {ring}
            –   unsat_bot : {ring}
            –   unsat_on  : [(upper , lower)]

    2.  Exploit the observation that one operator can fulfil a
        (top R S) *and* (on R X) simultaneously (analogous for *bot*):
            For every unsatisfied   (on a b)
                – if  a  is still in unsat_top  → pair them  (1 action instead of 2)
                – elif b is still in unsat_bot → pair them  (1 action instead of 2)

        Each successful pairing reduces the number of necessary actions by 1.

    3.  Let
            G  =  |unsat_top|  +  |unsat_bot|  +  |unsat_on|
            P  =  number of successful pairings
        The heuristic value is
                    h  =  G  –  P
        which is
            •   0  in every goal state (all three sets empty),
            •   strictly positive otherwise,
            •   finite for any reachable state since   G ≤ |goal-facts| .

    Although the heuristic is not admissible (it may underestimate when two
    still-unsatisfied literals cannot be achieved together in reality), it is
    inexpensive to compute and captures the essential *double-effect* nature
    of both operators, thereby guiding greedy best-first search with few
    expanded nodes.
    """

    def __init__(self, task):
        # -----------------------  parse the goal description  -----------------------
        self.top_goals = {}    # ring -> stick
        self.bot_goals = {}    # ring -> stick
        self.on_goals = []     # list of (upper , lower)

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                ring, stick = args
                self.top_goals[ring] = stick
            elif pred == "bot":
                ring, stick = args
                self.bot_goals[ring] = stick
            elif pred == "on":
                upper, lower = args
                self.on_goals.append((upper, lower))

    # ------------------------------------------------------------------------------

    def __call__(self, node):
        state = node.state

        # --------  collect all unsatisfied goal literals for the given state -------
        unsat_top = {
            r for r, s in self.top_goals.items()
            if f"(top {r} {s})" not in state
        }

        unsat_bot = {
            r for r, s in self.bot_goals.items()
            if f"(bot {r} {s})" not in state
        }

        unsat_on = [
            (a, b) for (a, b) in self.on_goals
            if f"(on {a} {b})" not in state
        ]

        # -----------------------------  greedy pairing  ----------------------------
        pairings = 0
        for (upper, lower) in unsat_on:
            if upper in unsat_top:
                pairings += 1
                unsat_top.remove(upper)
            elif lower in unsat_bot:
                pairings += 1
                unsat_bot.remove(lower)

        # total remaining unsatisfied goal literals
        remaining = len(unsat_top) + len(unsat_bot) + len(unsat_on)

        # heuristic estimate = unsatisfied goals – savings due to pairings
        return remaining - pairings
