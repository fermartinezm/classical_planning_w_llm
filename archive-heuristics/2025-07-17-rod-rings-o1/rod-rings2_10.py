from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions needed to achieve all goal conditions
    in the rod-rings domain. Each unsatisfied goal fact contributes a small, constant
    cost to the total heuristic value.

    Assumptions
    -----------
    1. Each fact in the rod-rings goal describes a ring arrangement, such as (top r1 s2)
       or (on r3 r5).
    2. Each action typically repositions a pair of rings, but multiple rearrangements
       might be needed overall.

    Heuristic Initialization
    ------------------------
    - We store all goal facts from the planning task as relevant conditions that must
      be satisfied in the final state.
    - Since the domain has no static facts relevant for the cost, no additional data
      structures are extracted in this phase.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Examine the current state to see which goal facts are already satisfied.
    2. For each goal fact that is missing in the current state, add a constant penalty
       (e.g., 2) to the heuristic cost.
       - The idea is that repositioning a ring to satisfy a goal fact normally
         requires some manipulation and might also require ensuring a ring is
         “holding” first.
    3. If all goal facts are satisfied (no missing goal fact), the heuristic
       is 0, indicating the goal state.
    """

    def __init__(self, task):
        """
        Extract goal facts from the provided planning task.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Not used here, but available if needed.

    def __call__(self, node):
        """
        Compute the heuristic value by counting how many goal facts
        are unsatisfied and multiplying by a constant factor.
        """
        state = node.state

        # Count unsatisfied goal facts.
        missing_goals = sum(1 for g in self.goals if g not in state)

        # If no goals are missing, return 0 (we are in a goal state).
        if missing_goals == 0:
            return 0

        # Otherwise, estimate cost as 2 actions per missing goal.
        return missing_goals * 2