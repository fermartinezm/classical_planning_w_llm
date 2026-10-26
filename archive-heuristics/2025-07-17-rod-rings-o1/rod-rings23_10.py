from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses
    and splitting the string. For example, '(top r2 s1)' becomes
    ['top', 'r2', 's1'].
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary:
    A simple domain-dependent heuristic for the rod-rings domain. This heuristic
    estimates the number of actions needed to satisfy all goal facts, based on
    how many goal facts are currently unsatisfied in the state.

    Assumptions:
    - Each ring can be at the top or bottom of a stick, on top of another ring,
      or held in the agent's hand.
    - Only two actions are available: put_top and put_bottom, both of which can
      simultaneously rearrange multiple rings.
    - Although this heuristic may not be admissible, it is cheap to compute and
      helps guide a greedy best-first search toward the goal.

    Heuristic Initialization:
    - Stores all goal facts and the static facts of the domain (though in this
      simple approach, static facts are unused).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if all goal facts are satisfied in the current state:
       - If yes, return 0 because the goal is already achieved.
    2. Otherwise, for each fact required by the goal that is not present in
       the current state, count it as unsatisfied.
    3. Each unsatisfied fact contributes a constant amount to the heuristic.
       - In this implementation, we multiply that count by 2 to reflect that
         often more than one ring relationship must be rearranged.
    4. The final heuristic value is 2 * (number of unsatisfied goal facts).

    This simple strategy yields a finite heuristic value for solvable states
    and returns 0 only in a goal state, thereby satisfying the basic requirements
    for a search-based heuristic.
    """

    def __init__(self, task):
        """
        :param task: The planning task, containing the goals
                     and any static facts (unused in this heuristic).
        """
        # Store all goal facts and static facts from the task
        self.goals = set(task.goals)
        self.static = task.static  # Not directly used by this heuristic

    def __call__(self, node):
        """
        Compute a numeric estimate of the remaining effort to satisfy the
        goal conditions for the rod-rings domain.

        :param node: A search node containing a state (frozenset of facts).
        :return: An integer heuristic value: 0 if the node is a goal state,
                 otherwise 2 × (number of unsatisfied goal facts).
        """
        state = node.state

        # If all goals are satisfied, return 0
        if self.goals.issubset(state):
            return 0

        # Count how many goal facts are not yet satisfied
        unsatisfied_count = sum(1 for fact in self.goals if fact not in state)

        # Multiply by 2 to reflect that re-arranging rings typically
        # requires multiple put_top/put_bottom actions
        return 2 * unsatisfied_count