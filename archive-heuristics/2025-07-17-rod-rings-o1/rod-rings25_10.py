from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the rod-rings domain that returns an estimated
    cost to transform the current state into a goal state. This heuristic simply
    counts how many goal facts are unsatisfied in the current state, and uses that
    number as the cost estimate.

    Assumptions
    -----------
    - The rod-rings domain goals typically involve ring positions on sticks 
      and relative positioning of rings (on, top, bot).
    - No ring is held at the goal state. (Though the domain might not explicitly 
      require this, most example problems do not specify a ring to be held in the goal.)
    - Each unsatisfied goal fact is taken to contribute a constant cost.

    Heuristic Initialization
    ------------------------
    - Goals are extracted from the provided task.
    - Static facts (if any) can be parsed here. However, in rod-rings, they are typically
      empty or not essential for this simple counting approach.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are present in the current state, return 0.
       (We already satisfy the goal.)
    2. Otherwise, iterate over each goal fact:
       a. Check if it is satisfied in the current state (i.e., it is in the set of
          state facts).
       b. If not satisfied, add 1 to the heuristic count.
    3. Return the total count of unsatisfied goals as the heuristic value.

    This simple method is a non-admissible but pragmatically efficient heuristic.
    """

    def __init__(self, task):
        """
        Store the goal facts and static facts for potential use.
        """
        self.goals = task.goals
        self.static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        :param node: A search node containing the current state.
        :return: An integer heuristic value estimating the cost to reach the goal.
        """
        state = node.state

        # If we already meet all goal facts, the heuristic is zero.
        if self.goals.issubset(state):
            return 0

        # Otherwise, add 1 for each goal fact not satisfied in the current state.
        return sum(1 for g in self.goals if g not in state)