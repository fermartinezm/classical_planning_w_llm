from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic computes an estimate of the number of actions required to achieve
    the goal configuration of rings and sticks in the rod-rings domain. Each unsatisfied
    goal fact contributes a fixed penalty to the heuristic value.

    Assumptions
    -----------
    • Every goal condition that appears in the domain (e.g., (top r s), (bot r s), (on r1 r2)) 
      must be satisfied.  
    • Each action in this domain has unit cost.  
    • A single action may satisfy multiple goal facts simultaneously, but for simplicity we
      assign a cost of 1 per unsatisfied fact.  
    • The domain is solvable from the given initial states (no infinite heuristic values).  

    Heuristic Initialization
    -------------------------
    • Store all goal facts from the task.  
    • Since there are no relevant static facts for rod-rings in this domain, there is no 
      usage of static facts for computing the heuristic.  

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify whether the state is already a goal state by checking if all goal facts 
       appear in the current state.  
       - If so, the heuristic value is 0.  
    2. Otherwise, count how many goal facts are missing from the current state.  
    3. Return that count as the heuristic value (i.e., number of unsatisfied goals).  
       - Each missing fact is assumed to require approximately one action to fix, giving 
         an integer estimate of the distance to goal.  
    """

    def __init__(self, task):
        """
        Initialize the rod-rings heuristic.

        Parameters
        ----------
        task : Task
            The planning task, which includes:
            - task.goals : set of goal facts
            - task.static: set of static facts (unused in this heuristic)
        """
        self.goals = task.goals
        # We do not use any static facts here, but if needed, we could parse them as well.
        static_facts = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        Parameters
        ----------
        node : Node
            A search node containing the current state (node.state).

        Returns
        -------
        int
            The estimated number of actions required to reach the goal from the current state.
        """
        state = node.state

        # If all goals are satisfied, return 0.
        if self.goals.issubset(state):
            return 0

        # Count unsatisfied goal facts.
        unsatisfied_goals = len(self.goals - state)

        return unsatisfied_goals