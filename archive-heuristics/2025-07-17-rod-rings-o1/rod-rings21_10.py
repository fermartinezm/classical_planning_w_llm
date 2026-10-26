from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting.
    Example: '(top r1 s1)' -> ['top', 'r1', 's1']
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - fact: The complete fact as a string, e.g., "(on r2 r4)".
    - args: The expected pattern (wildcards '*' allowed).
    Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This class provides a domain-dependent heuristic function for the rod-rings
    domain. Its primary goal is to estimate the number of actions required to
    achieve all the goal conditions involving rings' positions on the sticks or
    on other rings.

    Assumptions
    -----------
    1. Each ring can only be held if there is no other ring on it (as enforced by
       the domain logic).
    2. The domain actions, "put_top" and "put_bottom," implicitly pick up one ring
       and place another ring, making it unnecessary to have a separate "pick"
       action.
    3. This heuristic is not guaranteed to be admissible; it tries to provide a
       guidance that reduces the search effort.

    Heuristic Initialization
    ------------------------
    - Extract all goal facts from the task that mention rings: "on" (?top ring ?bot ring),
      "top" (?ring ring ?stick stick), or "bot" (?ring ring ?stick stick).
    - Store these goals per ring in a dictionary so that we can quickly check
      how many conditions each ring still needs to fulfill.
    - Any static facts in the domain are also retrieved but are not strictly used
      by this simple heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goal facts, return 0.
    2. Otherwise, for each ring mentioned in the goals:
       - Count how many ring-related goal facts are not satisfied in the current state.
       - Estimate the actions needed to fix those unsatisfied facts. For simplicity,
         this heuristic assumes each missing fact costs 2 actions:
         (a) rearranging another ring or picking up the ring, and
         (b) placing it at the correct position.
    3. Sum these costs for all rings. That total is returned as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by:
        - Capturing the set of goal facts (focusing on ring-related goals).
        - Capturing any static facts from the task (though not strictly used
          by this heuristic).
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build a mapping: ring -> set of goal facts involving that ring.
        self.ring_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate in {"on", "top", "bot"}:
                # "on ?top ?bot", "top ?r ?s", "bot ?r ?s"
                involved_ring_indices = [i for i, p in enumerate(parts[1:], start=1)
                                         if p.startswith("r")]  # ring objects start with 'r' in examples
                for idx in involved_ring_indices:
                    ring = parts[idx]
                    self.ring_goals.setdefault(ring, set()).add(goal)

    def __call__(self, node):
        """
        Evaluate how many actions might be needed to satisfy all ring-related goals.
        Each unsatisfied ring-goal fact is counted, multiplied by an approximate cost.
        """
        state = node.state

        # Quick check: if all goals are satisfied, return 0 immediately.
        if self.goals <= state:
            return 0

        total_cost = 0
        for ring, goals_for_ring in self.ring_goals.items():
            # Count how many of this ring's goals are not satisfied.
            missing_facts = sum(1 for g in goals_for_ring if g not in state)
            # Simple formula: each missing fact ~ 2 actions.
            total_cost += missing_facts * 2

        return total_cost