from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    For example, '(on r2 r4)' gets split into ['on', 'r2', 'r4'].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., '(top r2 s1)'.
    - args: A sequence of expected parts for matching, wildcards '*' allowed.
    - Returns True if the fact matches the pattern, otherwise False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic provides an approximate number of actions required to achieve
    all goal conditions in the rod-rings domain. It does so by counting how many
    goal facts are not yet satisfied in the current state.

    Assumptions
    -----------
    - There is at most one ring being held by the agent at any time.
    - An action (put_top or put_bottom) rearranges the current rod-rings configuration
      and may fix or disrupt multiple relationships at once.
    - The heuristic does not need to be admissible, only to guide the search effectively.

    Heuristic Initialization
    -------------------------
    - We store the goal facts from the task.
    - We also store any static facts from the domain, although in this domain they might be empty.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if all goal facts are already satisfied in the current state. If so, return 0.
    2. Otherwise, count how many goal facts are currently unsatisfied (missing) in the state.
    3. Each missing goal fact is assumed to require at most two rearrangement actions
       (though some may need fewer or more depending on the exact situation).
    4. Return the total estimated cost as "2 × [number of missing goal facts]".
    """

    def __init__(self, task):
        """
        Initialize the rod-rings heuristic.

        Parameters
        ----------
        task : Task
            A STRIPS planning task containing domain operators, initial state, and goals.
        """
        self.goals = task.goals
        # Store static facts if needed (though rod-rings may have none).
        self.static = task.static

    def __call__(self, node):
        """
        Compute an estimated cost to reach the goal configuration
        from the current state in the rod-rings domain.

        Parameters
        ----------
        node : A* node or any structure with a 'state' attribute
            Encapsulates the current state (as a frozenset of PDDL facts).

        Returns
        -------
        int
            An integer representing the estimated number of actions needed,
            based on how many goal conditions are not yet satisfied.
        """
        current_state = node.state

        # If the current state already satisfies all goal conditions, return 0.
        if self.goals.issubset(current_state):
            return 0

        # Count how many goal facts are missing.
        missing_goals = sum(1 for g in self.goals if g not in current_state)

        # Estimate a cost of 2 actions per missing goal fact.
        return 2 * missing_goals