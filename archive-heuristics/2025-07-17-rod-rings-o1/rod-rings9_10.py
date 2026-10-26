from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove the surrounding parentheses from a fact string and split the result by spaces.
    For example, '(top r2 s1)' -> ['top', 'r2', 's1'].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (with possible wildcards).
    For example, match('(top r2 s1)', 'top', '*', '*') -> True.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This is a domain-dependent heuristic for the rod-rings domain. It provides
    a rough estimate of how many additional moves (put_top or put_bottom) are
    needed to achieve the goal state based on whether each ring involved in the
    goal has satisfied all of its goal conditions.

    Assumptions
    -----------
    1. Each ring can appear in zero or more goal facts (top, bot, on).
    2. A single rod-rings action can place a ring in its correct top/bottom
       position and also establish the correct (on top) relationship in one step,
       provided the planner is already holding that ring.
    3. This heuristic treats each ring’s goals collectively. If a ring needs
       multiple facts satisfied (e.g. “top” plus “on”), we expect one put_top
       or put_bottom action can generally fix them simultaneously.

    Heuristic Initialization
    -------------------------
    1. Store the complete list of goal facts from the task.
    2. For each goal fact, if it involves a ring (with predicates "top", "bot",
       or "on"), group that fact under that particular ring. This results in
       a mapping from ring -> set of goal facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if all goal facts are satisfied in the current state. If so,
       return 0 (goal reached).
    2. Otherwise, for each ring in our ring-to-goals mapping:
       a. Retrieve all goal facts for that ring.
       b. If any of these facts are missing in the current state, we assume
          we need at least one action to fix them. Hence, add 1 to the cost
          for that ring.
    3. Sum up the cost contributions from all rings and return that sum as
       the heuristic value.
    """

    def __init__(self, task):
        """
        Extract goal facts and group them by ring. Static facts are stored
        but not used in this particular heuristic.
        """
        self.goals = task.goals
        self.static = task.static  # Not used in this heuristic, but stored for completeness

        # Map of ring -> set of goal facts pertaining to that ring
        self.ring_goals = {}
        for goal_fact in self.goals:
            predicate, *args = get_parts(goal_fact)
            # Only consider ring-related predicates: top, bot, on
            if predicate in ["top", "bot", "on"]:
                # For top/bot: ring is args[0]
                # For on: ring is args[0] (the "top" ring in "on ?top ?bot")
                ring = args[0]
                if ring not in self.ring_goals:
                    self.ring_goals[ring] = set()
                self.ring_goals[ring].add(goal_fact)

    def __call__(self, node):
        """
        Compute the heuristic value for the current state. If the goal is reached,
        return 0. Otherwise, for each ring mentioned in the goal, check if all
        the ring's goal facts are satisfied; if not, add 1 for that ring.
        """
        state = node.state

        # If all goals are satisfied, return 0 immediately
        if self.goals <= state:
            return 0

        cost = 0
        # For each ring that appears in the goals
        for ring, goal_facts in self.ring_goals.items():
            # If any fact for this ring is not in the current state, we assume 1 action is needed
            if not goal_facts <= state:
                cost += 1

        return cost