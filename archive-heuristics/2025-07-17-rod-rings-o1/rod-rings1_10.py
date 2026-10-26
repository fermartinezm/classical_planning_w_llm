from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing surrounding parentheses
    and splitting on whitespace.

    Example:
        If fact = '(on r2 r1)', then get_parts(fact) = ['on', 'r2', 'r1'].
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (supporting wildcards).

    Args:
        fact: The complete fact as a string, e.g., "(on r2 r1)".
        args: The expected pattern, where wildcards (*) are allowed.

    Returns:
        True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary:
        This is a domain-dependent heuristic for the 'rod-rings' domain. It
        estimates the number of actions required to achieve the goal state
        from the current state, balancing a fair estimate with efficient
        computation. 

    Assumptions:
        - The domain contains rods (sticks) and rings.
        - Placing a ring in its correct position typically requires a few
          actions (put_top/put_bottom).
        - The heuristic does not have to be admissible, so it may overestimate
          the distance to the goal.

    Heuristic Initialization:
        - We store the goal facts from the planning task. 
        - We note that there are no static facts needed for special data 
          structures in this domain.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is already a goal state. If yes, return 0.
        2. Otherwise, count how many goal facts are currently unsatisfied in
           the state.
        3. Each unsatisfied goal fact is assigned a small constant cost
           (e.g., 2) to reflect the approximate effort of repositioning rings.
        4. Sum these costs to produce the heuristic value.
        5. Return this sum as the estimate of actions required to solve
           the task.
    """

    def __init__(self, task):
        """
        Initializes rod_ringsHeuristic by capturing the goal facts from
        the planning task. The 'static' facts set is present but not used
        due to this domain's simplicity.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Not used, but stored for completeness

    def __call__(self, node):
        """
        Computes the heuristic value for the rod-rings domain.

        Args:
            node: A search node containing the current state (frozenset of facts).

        Returns:
            An integer representing the estimated number of actions required
            to reach the goal from the current state.
        """
        state = node.state

        # If all goals are satisfied, the heuristic is 0.
        if self.goals.issubset(state):
            return 0

        # Otherwise, count how many goal facts are missing.
        missing_goals = sum(1 for g in self.goals if g not in state)

        # Assign a small constant cost (e.g., 2) per missing goal fact.
        # This serves as a rough estimate of the actions required
        # (since placing a ring often involves at least two steps).
        return 2 * missing_goals