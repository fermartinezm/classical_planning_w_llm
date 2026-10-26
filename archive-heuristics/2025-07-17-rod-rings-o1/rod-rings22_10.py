from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses
    and splitting the string.
    Example:
        fact = '(on r4 r5)'
        returns ['on', 'r4', 'r5']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - fact: The complete fact as a string, e.g., "(on r4 r5)".
    - args: The expected pattern (wildcards `*` allowed).
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rod_ringsHeuristic(Heuristic):
    """
    Summary:
        This heuristic estimates how many “ring configurations” are still incorrect 
        relative to the goal state in the rod-rings domain. The domain-specific insight 
        is that each ring’s final placement typically involves at most one 
        put_top/put_bottom action, which can fix both “on ring” and “top/bot on stick” 
        if done correctly. Hence, tracking whether each ring’s goal conditions are 
        satisfied or not produces a meaningful heuristic value.

    Assumptions:
        1. Each ring in a goal condition either must be on another ring 
           (e.g., "(on r2 r4)") or must be at the top/bottom of a particular stick 
           (e.g., "(top r2 s1)", "(bot r5 s2)").
        2. At most one ring is held at a time, but typically the goals require 
           rings to end up "on" or "top"/"bot" a stick (i.e., not held).
        3. Solvable problem instances will have well-defined goal conditions for 
           rings and sticks.

    Heuristic Initialization:
        - We store for each ring all the goal facts that mention it. For example, 
          if the goal states "(top r2 s1)" and "(on r2 r4)", both facts are attached 
          to ring "r2".
        - No static facts need to be extracted for typical rod-rings problems.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the entire goal set is satisfied by the current state. 
           If so, heuristic = 0.
        2. Otherwise, for each ring mentioned in the goals:
           a. Identify all the goal facts for that ring (e.g., top/bot requirement, 
              on-relationship).
           b. Check if these specific facts are satisfied in the current state.
           c. If all facts for that ring are satisfied, cost contribution for that 
              ring is 0. Otherwise, cost contribution is 1.
        3. Sum the cost over all rings. This sum is the returned heuristic value.
           Thus, each ring’s misplacement contributes 1 to the heuristic, 
           reflecting that typically one put_top/put_bottom action can fix its final 
           configuration.
    """

    def __init__(self, task):
        """
        Store the goal conditions by ring. This allows us to check, for each ring,
        whether its goal facts are fulfilled or not.
        """
        self.goals = task.goals
        # Map from ring -> set of goal facts (strings) that involve that ring.
        self.ring_goals = {}

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            # Predicates of interest: on ?top ?bot, top ?r ?s, bot ?r ?s, holding ?r
            # We attach each goal fact to the "primary" ring mentioned.
            if predicate == "on":
                # "on rTop rBot"
                ring_top = parts[1]
                self.ring_goals.setdefault(ring_top, set()).add(goal_fact)
            elif predicate in ("top", "bot", "holding"):
                ring = parts[1]
                self.ring_goals.setdefault(ring, set()).add(goal_fact)
            # If some fact doesn't match these forms, we ignore it (unlikely in rod-rings).

    def __call__(self, node):
        """
        Calculate the heuristic value for the current state. 
        The heuristic is 0 if the goals are fully satisfied. Otherwise, 
        for each ring that is not in its proper final configuration, add 1 to the total.
        """
        state = node.state

        # 1) If the entire goal set is satisfied, return 0 immediately.
        if self.goals <= state:
            return 0

        total_cost = 0

        # 2) For each ring, check its relevant goal facts.
        for ring, goal_facts_for_ring in self.ring_goals.items():
            # Check if each fact in goal_facts_for_ring is in the current state.
            # If any are missing, increment cost by 1 for this ring.
            if not goal_facts_for_ring.issubset(state):
                total_cost += 1

        return total_cost