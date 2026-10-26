from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def parts(fact: str):
    """Return the token list of a PDDL fact string (without the brackets)."""
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the **rod-rings** domain.  The heuristic
    tries to predict how many ``put_top``/``put_bottom`` actions are still
    required to satisfy the goal description of the current planning task.

    Assumptions
    -----------
    1.  Exactly one ring is *always* held (this is the case in all benchmark
        instances that accompany the domain).
    2.  Every ring resides either in the hand or on exactly one stick; the
        stick configuration forms a single linear chain that starts with
        the ``top`` ring and is continued by the ``on`` predicates down to the
        ``bot`` ring.
    3.  Goal formulae contain only the predicates **top**, **bot** and **on**.
        No goal references ``holding``.
    4.  The heuristic is *informed* but **not admissible** – it purposely
        over-estimates in order to reduce the number of expanded nodes in a
        Greedy Best-First Search.

    Heuristic Initialization
    ------------------------
    While constructing the heuristic object we
        • store the individual goal facts, split into three dictionaries  
          (top-goals, bot-goals, on-goals) for quick access,  
        • remember the *raw* goal fact set in order to recognise goal states
          immediately (cost 0).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search node, the heuristic performs the following steps:

    1.  **Analyse the current state**
        • identify the ring that is *held*;  
        • build for each stick the *stack list* `[top, … , bottom]`  
          by following the chain of ``on`` predicates;  
        • for every ring memorise its
          – location (`"hand"` or stick-name) and  
          – depth on that stick (index 0 = top).

    2.  **Evaluate ON goals**
        • if an ``(on r1 r2)`` goal is already satisfied, add 0;  
        • if `r1` and `r2` are on the *same* stick but have
          `k = depth(r2) − depth(r1) − 1 ≥ 0` intermediate rings,
          at least *k + 1* rotations are needed → add `k + 1`;  
        • otherwise the two rings are on different sticks / one is in hand –
          this certainly needs *two* rotations (one to grab `r1`, one to drop
          it onto `r2`) → add 2.

    3.  **Evaluate TOP goals**
        • if already satisfied, add 0;  
        • if the wanted ring is *in hand*, one ``put_top`` suffices → add 1;  
        • if the ring is on the *correct* stick at depth `d > 0`,
          we need *d* ``put_bottom`` rotations to free it **plus one**
          ``put_top`` to re-insert it → add `d + 1`;  
        • if it resides on another stick, one additional rotation is required
          (to move it onto the correct stick) → add `d + 2`.

    4.  **Evaluate BOT goals** (handled symmetrically to TOP goals).

    5.  **Return the accumulated cost** – by construction this is  
        • `0` in any goal state, and  
        • finite for every reachable state.

    Although coarse-grained, the heuristic distinguishes “easy” from “hard”
    situations far better than a simple goal-counting heuristic and thereby
    reduces the overall search effort.
    """

    def __init__(self, task):
        # Split the goal facts for quicker access
        self.goal_facts = task.goals
        self.goal_top = {}     # ring -> stick
        self.goal_bot = {}     # ring -> stick
        self.goal_on = {}      # ring_above -> ring_below

        for fact in task.goals:
            pred, *args = parts(fact)
            if pred == "top":
                ring, stick = args
                self.goal_top[ring] = stick
            elif pred == "bot":
                ring, stick = args
                self.goal_bot[ring] = stick
            elif pred == "on":
                upper, lower = args
                self.goal_on[upper] = lower

    # ---------------------------------------------------------------------- #
    #  Helper functions                                                       #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _match(fact, pred, *args):
        """fnmatch-based convenience wrapper."""
        tokens = parts(fact)
        return tokens[0] == pred and all(fnmatch(t, p) for t, p in zip(tokens[1:], args))

    def _build_stacks(self, state):
        """
        Reconstruct every stick’s stack:

        returns
        -------
        stacks : dict   stick -> [ring_top , … , ring_bottom]
        pos    : dict   ring  -> ("hand", None)  OR  (stick, depth)
        held   : str    the ring in hand
        """
        on_map = {}
        stacks = {}
        pos = {}

        held = None
        for fact in state:
            pred, *args = parts(fact)
            if pred == "holding":
                held = args[0]
            elif pred == "on":
                upper, lower = args
                on_map[upper] = lower
            elif pred == "top":
                ring, stick = args
                stacks[stick] = [ring]          # initialise stack with top ring
            elif pred == "bot":
                ring, stick = args
                # bottom fact not used to build the list but stored below
                pass

        # Follow the ON-links to complete every stack
        for stick, lst in stacks.items():
            current = lst[0]
            while current in on_map:             # follow chain downward
                current = on_map[current]
                lst.append(current)

        # fill pos-table
        for stick, lst in stacks.items():
            for depth, ring in enumerate(lst):
                pos[ring] = (stick, depth)
        if held:
            pos[held] = ("hand", None)

        return stacks, pos, held

    # ---------------------------------------------------------------------- #
    #  Main call                                                              #
    # ---------------------------------------------------------------------- #
    def __call__(self, node):
        state = node.state
        # Fast exit for goal states
        if self.goal_facts <= state:
            return 0

        stacks, pos, held = self._build_stacks(state)
        cost = 0

        # --- handle ON goals ------------------------------------------------
        for upper, lower in self.goal_on.items():
            if f"(on {upper} {lower})" in state:
                continue  # already satisfied

            # identify current positions
            p_up  = pos.get(upper, ("hand", None))
            p_low = pos.get(lower, ("hand", None))

            # same stick and both on stick?
            if p_up[0] == p_low[0] and p_up[0] != "hand":
                dist = p_low[1] - p_up[1] - 1
                if dist >= 0:
                    cost += dist + 1  # 0 rings between → 1; else >1 accordingly
                else:
                    # upper is *not* above lower → needs at least 2 actions
                    cost += 2
            else:
                cost += 2  # different sticks / one in hand

        # --- handle TOP goals ----------------------------------------------
        for ring, stick in self.goal_top.items():
            if f"(top {ring} {stick})" in state:
                continue
            loc, depth = pos.get(ring, ("hand", None))

            if loc == "hand":
                cost += 1  # one put_top
            elif loc == stick:
                cost += depth + 1  # depth put_bottoms + 1 put_top
            else:
                cost += (depth + 2) if depth is not None else 2

        # --- handle BOT goals ----------------------------------------------
        for ring, stick in self.goal_bot.items():
            if f"(bot {ring} {stick})" in state:
                continue
            loc, depth = pos.get(ring, ("hand", None))

            if loc == "hand":
                cost += 1  # one put_bottom
            elif loc == stick:
                # number of rings below the desired ring
                depth_from_bottom = len(stacks[stick]) - depth - 1
                cost += depth_from_bottom + 1
            else:
                cost += (depth + 2) if depth is not None else 2

        return cost
