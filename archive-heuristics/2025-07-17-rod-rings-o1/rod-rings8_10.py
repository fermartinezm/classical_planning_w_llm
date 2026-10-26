from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the rod-rings domain. The heuristic
    estimates how many additional actions are required to satisfy the goal
    conditions, counting how many goal facts are not yet satisfied in the
    current state.

    Assumptions
    -----------
    - Each action can potentially fix one or more unsatisfied goal facts.
    - This heuristic is not necessarily admissible; it is intended to be
      efficient and guide a greedy search toward the goal.

    Heuristic Initialization
    -------------------------
    - We store the set of goal facts from the task. The rod-rings domain does
      not use static facts in a way that requires special processing, so static
      information is unused here.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Obtain the current state of the search node.
    2. Check if all goal facts are present in the current state:
       - If so, the heuristic value is zero.
    3. Otherwise, count how many goal facts are not yet satisfied.
    4. Use this count as the heuristic value, thereby providing a simple
       estimate of how many more actions are needed.

    """

    def __init__(self, task):
        """
        Extracts the goals and static facts from the task. In this domain, static
        facts do not provide additional structure needed for the heuristic, so we
        only store the goals.

        :param task: The planning task, containing goals and static facts.
        """
        self.goals = task.goals
        # Store static facts if needed in the future (unused for now).
        self.static_facts = task.static

    def __call__(self, node):
        """
        Computes a heuristic value based on how many goal facts are not yet
        satisfied in the current state.

        :param node: A search node containing the current state.
        :return: An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # If all goals are satisfied, return 0 immediately.
        if self.goals.issubset(state):
            return 0

        # Otherwise, count unsatisfied goal facts.
        missing_goals_count = sum(1 for g in self.goals if g not in state)
        return missing_goals_count