from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact: str) -> list:
    """
    Extract the components of a PDDL fact by removing parentheses
    and splitting the string on whitespace.

    Example:
    If fact = "(holding r4)", then get_parts(fact) -> ["holding", "r4"].
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the rod-rings domain.

    Summary:
    This heuristic estimates the number of actions needed by counting
    how many goal facts remain unsatisfied. Each domain action can fix
    at least one unsatisfied goal fact, so the count of unsatisfied goals
    provides a coarse but effective estimate of the actions required.

    Assumptions:
    - The goal is fully defined by the presence of certain predicates (on, top, bot, holding).
    - Each unsatisfied goal predicate likely requires at least one rod-rings action to achieve.
    - No static facts are required in this domain (i.e., there is no extra domain information).

    Heuristic Initialization:
    - We store all goal facts from the task in a set.
    - Although static facts exist in other domains, this domain does not conventionally use them.

    Step-By-Step Thinking for Computing Heuristic:
    1) Let G be the set of goal facts in the task.
    2) For the current state, count how many goal facts from G are already satisfied.
    3) The heuristic value is the number of unsatisfied goals: len(G) - number_satisfied.
       - This ensures a value of 0 when all goals are satisfied (goal state).
       - Otherwise, the heuristic is positive and directly correlates with how many goal facts remain unsatisfied.
    """

    def __init__(self, task):
        """
        Initialize the rod-rings heuristic by extracting goal facts and
        (if needed) static facts from the task.
        """
        self.goal_facts = frozenset(task.goals)  # Store all the goal facts.

        # Static facts are available but not used for this domain-specific approach.
        self.static_facts = task.static

    def __call__(self, node):
        """
        Calculate the heuristic value as the number of unsatisfied goal facts.
        """
        state = node.state
        # Count how many of the goal facts are satisfied by the current state.
        satisfied_count = sum(1 for goal in self.goal_facts if goal in state)

        # The heuristic is the number of unsatisfied goals.
        return len(self.goal_facts) - satisfied_count