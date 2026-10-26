from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """
    Helper that splits a fact into its components.

    Example
    -------
    >>> get_parts("(top r1 s1)")
    ['top', 'r1', 's1']
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    Domain–dependent heuristic for the *rod-rings* domain.  The heuristic
    returns an **informed but inadmissible** estimate of how many
    `put_top` / `put_bottom` actions are still necessary to satisfy the
    remaining goal facts.

    Assumptions
    -----------
    1.  Every stick always carries exactly two rings (one `top`, one `bot`)
        and there is exactly one ring in the robot’s hand (`holding`).
    2.  A single action may simultaneously establish up to *two* goal
        facts –– e.g. `put_top` can make both `(top X S)` **and**
        `(on  X Y)` true for the same stick.
    3.  Moving a ring that is **already held** to its desired position
        costs one action, whereas moving a ring that currently sits on a
        stick takes *at least* two actions (first pick it up, then put it
        down somewhere else).

    Heuristic Initialization
    ------------------------
    While being constructed the heuristic separates the goal facts into
    convenient structures:

    • `goal_top[stick]  -> ring`        — desired top rings  
    • `goal_bot[stick]  -> ring`        — desired bottom rings  
    • `goal_on [ring]   -> ring`        — desired *top–bottom* pairs  
      (keys are the **top** rings)

    These tables are used in every call to quickly assess how far the
    current state is from the goal.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Parse the current *search node*  
        – identify the ring that is `holding`, the current `(top …)` and
        `(bot …)` rings of every stick, and map each ring to its concrete
        position (`hand`, `top`, `bottom`).

    2.  Walk over every stick that appears in a *top* goal:  
        a)  If **both** the required `(top R S)` *and* its associated
            bottom condition (`(on R X)` *or* `(bot X S)`) are already
            satisfied → 0 cost.  
        b)  If exactly one of them is missing → 1 action is considered
            sufficient.  
        c)  If neither is satisfied  
            – and the wanted top ring is currently *held* → 1,  
            – otherwise                           → 2 (we must first grab
              that ring and then place it).

    3.  Deal with remaining goal facts that were not covered in step 2  
        (i.e. stand-alone `on` facts or `bot` facts for sticks that do not
        have a `top` goal):  
        – satisfied → 0  
        – unsatisfied & **held ring** is the one that must be positioned
          → 1  
        – otherwise → 2.

    4.  Return the accumulated cost.  
        The value is always *finite* for reachable states and exactly `0`
        for goal states because the very first check compares the complete
        goal set with the current state.
    """

    def __init__(self, task):
        # Store the complete set of goal facts for the quick-goal test
        self.goal_facts = task.goals

        # Extract structured goal information
        self.goal_top = {}  # stick  -> ring
        self.goal_bot = {}  # stick  -> ring
        self.goal_on = {}   # topRing -> bottomRing

        for fact in task.goals:
            pred, *args = get_parts(fact)
            if pred == "top":
                ring, stick = args
                self.goal_top[stick] = ring
            elif pred == "bot":
                ring, stick = args
                self.goal_bot[stick] = ring
            elif pred == "on":
                top_ring, bottom_ring = args
                self.goal_on[top_ring] = bottom_ring

    # --------------------------------------------------------------------- #
    #                        heuristic evaluation                           #
    # --------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state

        # 0 ––> we are done
        if self.goal_facts <= state:
            return 0

        # ---------------------------------------------------------------- #
        #                   1.   analyse the current state                 #
        # ---------------------------------------------------------------- #
        held_ring = None
        ring_pos = {}        # ring -> ('hand') | ('top', stick) | ('bot', stick)

        for fact in state:
            pred, *args = get_parts(fact)

            if pred == "holding":
                held_ring = args[0]
                ring_pos[held_ring] = ("hand",)

            elif pred == "top":
                ring, stick = args
                ring_pos[ring] = ("top", stick)

            elif pred == "bot":
                ring, stick = args
                ring_pos[ring] = ("bot", stick)

        # ---------------------------------------------------------------- #
        #                   2.   evaluate goal satisfaction                 #
        # ---------------------------------------------------------------- #
        cost = 0
        processed_sticks = set()

        # 2a) sticks that have an explicit top goal
        for stick, goal_top_ring in self.goal_top.items():
            processed_sticks.add(stick)

            # whether the TOP part is already correct
            top_ok = f"(top {goal_top_ring} {stick})" in state

            # check the required relation beneath the top ring -------------
            if goal_top_ring in self.goal_on:
                goal_bottom_ring = self.goal_on[goal_top_ring]
                bottom_ok = f"(on {goal_top_ring} {goal_bottom_ring})" in state
            elif stick in self.goal_bot:
                goal_bottom_ring = self.goal_bot[stick]
                bottom_ok = f"(bot {goal_bottom_ring} {stick})" in state
            else:
                # no requirement on what is below the top ring
                bottom_ok = True

            # --------- cost contribution ----------------------------------
            if top_ok and bottom_ok:
                continue                           # perfect – no cost
            elif top_ok or bottom_ok:
                cost += 1                          # one of them is wrong
            else:
                # both parts are wrong – optimistic tie-breaking:
                cost += 1 if held_ring == goal_top_ring else 2

        # 2b) stand-alone ON goals (whose top ring is *not* already covered)
        for top_ring, bottom_ring in self.goal_on.items():
            if top_ring in self.goal_top.values():
                continue  # already considered above

            if f"(on {top_ring} {bottom_ring})" in state:
                continue
            cost += 1 if held_ring == top_ring else 2

        # 2c) stand-alone BOT goals (sticks with no TOP goal)
        for stick, bottom_ring in self.goal_bot.items():
            if stick in processed_sticks:
                continue

            if f"(bot {bottom_ring} {stick})" in state:
                continue
            cost += 1 if held_ring == bottom_ring else 2

        return cost
