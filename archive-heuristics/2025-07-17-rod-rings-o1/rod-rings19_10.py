from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    For example, if fact == "(on r2 r1)", this function returns ["on", "r2", "r1"].
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic provides a domain-dependent estimate of the number of actions required
    to achieve the goal conditions for the rod-rings domain.

    Assumptions
    -----------
    1. Each goal fact (for example, '(on r2 r4)' or '(top r2 s1)') that is not satisfied
       likely requires at least one action to fix.
    2. The heuristic is not necessarily admissible, but it should help guide the search
       by reflecting how many ring placements remain misplaced.
    3. The planner's representation of the state includes all current rod-rings facts, and
       the "static" facts set is either empty or unused in this domain.

    Heuristic Initialization
    -------------------------
    - We store all goal facts from the given task in the constructor.
    - We also store the static facts, although in this domain they are typically empty.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state already satisfies all goal facts:
       - If yes, return 0.
    2. Otherwise, for each goal fact (e.g., '(on r2 r4)', '(top r3 s2)', etc.):
       - If that fact is not present in the current state, increment the cost by 1.
       - Each missing goal fact is taken as a proxy for at least one "put_top" or "put_bottom"
         action (or a related move) needed to correct the ring arrangement.
    3. Sum these increments to produce the heuristic value, returning it as an integer.

    This simple domain-dependent approach leverages the structure of the rod-rings domain
    (which primarily revolves around achieving correct "(on ? ?)" and "(top ? ?)" relationships)
    and proves effective for guiding a greedy best-first search without guaranteeing admissibility.
    """

    def __init__(self, task):
        """
        Store goal facts and static facts for potential use during heuristic calculation.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Potentially unused in this domain.

    def __call__(self, node):
        """
        Calculate a heuristic estimate of the number of actions needed to satisfy all goal facts.

        :param node: The current search node, whose state is a frozenset of facts.
        :return: An integer heuristic value.
        """
        state = node.state

        # If all goals are already satisfied, return 0.
        if self.goals.issubset(state):
            return 0

        # Otherwise, count how many goal facts are missing.
        cost = 0
        for goal_fact in self.goals:
            if goal_fact not in state:
                cost += 1

        return cost