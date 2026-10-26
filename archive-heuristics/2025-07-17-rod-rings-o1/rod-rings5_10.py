from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses
    and splitting the resulting string.
    For example, '(on r1 r2)' -> ['on', 'r1', 'r2'].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., "(on r1 r2)".
    - args: The expected pattern (wildcards '*' allowed).
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rod-rings domain.

    ----------------------------------------------------------------------------
    Summary
    ----------------------------------------------------------------------------
    This heuristic estimates how many rings are not yet in their goal positions.
    For each ring, we check whether all of the goal facts concerning that ring
    are already satisfied in the current state. If not, we add a small penalty
    (e.g., 2) to account for the actions needed to place it correctly.

    ----------------------------------------------------------------------------
    Assumptions
    ----------------------------------------------------------------------------
    1. Each ring's correct placement is specified by one or more goal facts
       (e.g., "(top rX sY)" or "(on rX rY)").  
    2. The goal may not mention every ring; any ring not mentioned is assumed
       not to require any specific placement (and thus contributes no cost).
    3. The domain allows at most one ring to be held at a time, but the heuristic
       only checks final placement of rings, ignoring intermediate constraints.
    4. This heuristic need not be admissible, only effectively guiding a
       greedy best-first search.

    ----------------------------------------------------------------------------
    Heuristic Initialization
    ----------------------------------------------------------------------------
    - The constructor gathers all goal facts and groups them by the ring(s)
      mentioned. For example, if the goal says "(on r2 r4)" and "(top r2 s1)",
      both facts are stored under ring r2.

    - Static facts are provided to the constructor but are not strictly needed
      for the rod-rings domain, so they remain unused.

    ----------------------------------------------------------------------------
    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------------------------------------
    1. Check for goal satisfaction. If the current state already satisfies all
       goal facts, return 0.
    2. For each ring mentioned in at least one goal fact:
       a. Retrieve the collection of goal facts for that ring.
       b. If all those facts are included in the current state, no additional
          actions are needed for that ring.
       c. Otherwise, add a fixed penalty (2) to the heuristic. This accounts
          for the likely need to reposition or re-stack that ring.
    3. Sum the penalties for all rings. This sum is the heuristic value
       indicating how many rings still need rearranging.
    """

    def __init__(self, task):
        """
        Initialize the rod-rings heuristic by:
         - Storing all goal facts for each ring.
         - Optionally reading static facts (though they are unused here).
        """
        self.goals = task.goals
        self.static_facts = task.static  # Unused in this heuristic.

        # A dictionary mapping each ring to the set of goal facts that mention it.
        self.ring_goals = {}

        for g in self.goals:
            parts = get_parts(g)
            predicate = parts[0]
            if predicate == "on":
                # (on ring1 ring2)
                ring1, ring2 = parts[1], parts[2]
                # ring1's correct placement depends on this 'on' fact
                self.ring_goals.setdefault(ring1, set()).add(g)

                # ring2 is also part of the goal, but 'on r1 r2' only
                # directly constrains r1. If the goal includes constraints
                # for r2, they will appear in separate facts (e.g. top/bot).
            elif predicate == "top":
                # (top ring stick)
                ring, _ = parts[1], parts[2]
                self.ring_goals.setdefault(ring, set()).add(g)
            elif predicate == "bot":
                # (bot ring stick)
                ring, _ = parts[1], parts[2]
                self.ring_goals.setdefault(ring, set()).add(g)
            # Other predicates, if any, would be handled similarly.

    def __call__(self, node):
        """
        Return an estimate of the number of actions required to achieve
        the goal from the current state of the node.
        """
        state = node.state

        # If the goal is already satisfied, heuristic = 0.
        if node.task.goal_reached(state):
            return 0

        cost = 0
        # For each ring in the ring_goals dictionary, check if its goal facts are satisfied.
        for ring, goal_facts in self.ring_goals.items():
            # If not all goal facts for this ring are satisfied, add a penalty.
            if not goal_facts.issubset(state):
                cost += 2

        return cost