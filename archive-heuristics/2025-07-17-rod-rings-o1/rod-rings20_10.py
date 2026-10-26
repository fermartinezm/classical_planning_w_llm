from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing its surrounding parentheses
    and splitting the string. For example:
      '(on r2 r1)' -> ['on', 'r2', 'r1']
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g. "(on r2 r1)".
    - args: The expected pattern (wildcards '*' allowed).
    
    Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This domain-dependent heuristic for the rod-rings domain estimates how many
    actions remain to achieve the configuration described by the goal facts.

    Assumptions
    -----------
    1) Each PDDL action in this domain rearranges exactly one ring from the 
       top or bottom of a stick to a new position. 
    2) Only one ring can be held at a time.
    3) For each goal fact (on, top, or bot), satisfying that fact is considered
       indicative of at least one needed rearrangement if it is not already 
       satisfied.

    Heuristic Initialization
    -------------------------
    1) The goal facts and static facts are extracted from the task.
    2) Static facts are stored for potential future extension, though the 
       supplied domain usually has no static facts.
    3) We do not precompute elaborate structures because the rod-rings domain
       can have variable ring-stacking, and a straightforward check of
       unsatisfied goal facts is sufficient to yield a guiding heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1) If the current state already satisfies all of the goal facts, the 
       heuristic is zero. 
    2) Otherwise, we count how many goal facts are not satisfied by the current
       state. 
    3) Each unsatisfied goal fact is assumed to require a non-trivial amount
       of rearranging. We multiply this count by a small constant factor 
       (for example, 2) to reflect that each ring may need to be moved or
       repositioned in multiple steps (due to the need to free rings or sticks).
    4) The resulting sum is returned as the heuristic value.
    """

    def __init__(self, task):
        """
        Store the goals and static facts for potential use.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Usually empty in rod-rings domain

    def __call__(self, node):
        """
        Return a heuristic value estimating how many actions remain to make 
        all goal facts true.
        """
        state = node.state
        
        # If all goals are already satisfied, return 0.
        if node.task.goal_reached(state):
            return 0

        # Count how many goal facts are unsatisfied.
        unsatisfied_goals = sum(1 for g in self.goals if g not in state)

        # Multiply by a small factor to reflect that rearranging one ring
        # often requires at least putting and taking in a single action pair.
        return unsatisfied_goals * 2