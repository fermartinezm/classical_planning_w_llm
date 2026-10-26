from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove surrounding parentheses from a fact string and split by whitespace.
    Example: "(holding r1)" -> ["holding", "r1"].
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
    name: rod-ringsHeuristic

    Summary
    -------
    This domain-dependent heuristic for the "rod-rings" domain estimates the number of
    actions needed to satisfy any missing goal literals. For each goal fact that is not
    present in the current state, a fixed cost is added to represent the approximate
    effort of picking up and placing rings.

    Assumptions
    -----------
    1. Only one ring can be held at a time.
    2. Each missing goal literal typically requires at least one or two movements of rings.
    3. The heuristic is not necessarily admissible, but aims to guide the search effectively.

    Heuristic Initialization
    ------------------------
    - Store all goal facts in self.goals.
    - Extract static facts (if any) for possible use. In this domain, there are no relevant
      static relationships to track, so this is left as a placeholder.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state satisfies all goal literals. If yes, return 0.
    2. Otherwise, for each goal literal that is missing from the state, add a constant cost.
       We use 2 as a simplification to represent the cost of picking up and repositioning
       a ring (or doing a bottom/top placement).
    3. Sum these costs to produce the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic with the task's goal facts and store any static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """
        Compute and return the heuristic value for the given node.
        """
        state = node.state

        # If the state already satisfies all goals, return 0.
        if self.goals.issubset(state):
            return 0

        # For each unsatisfied goal literal, add a fixed cost of 2.
        cost = 0
        for goal in self.goals:
            if goal not in state:
                cost += 2

        return cost