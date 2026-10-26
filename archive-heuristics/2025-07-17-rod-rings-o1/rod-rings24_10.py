from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    Example:
        fact = "(top r1 s1)"
        Returns ["top", "r1", "s1"]
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This is a domain-dependent heuristic for the rod-rings domain. It estimates
    the number of actions (put_top or put_bottom) required to achieve the goal
    state from the current state in a greedy best-first search. The heuristic
    is not necessarily admissible, but aims to remain computationally efficient
    while providing a good estimate.

    Assumptions
    -----------
    - Each goal condition in this domain is either (top ?ring ?stick) or (on ?ring ?other_ring).
    - The actions in rod-rings allow swapping rings between the robot's "holding" state
      and the top or bottom positions on sticks.
    - The domain typically requires each ring to be placed in a specified relationship
      with another ring or stick.

    Heuristic Initialization
    -------------------------
    - We store the set of goal facts from the given task.
    - No additional static facts are processed here, as the rod-rings domain has no
      specific invariants we need to parse (the :static set is typically empty).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal facts: for each ring, a goal fact might be (top r s) or (on r1 r2).
       Collect those goal facts directly from task.goals.
    2. For the current state, count how many of these goal facts are not satisfied (i.e.,
       not present in the state).
    3. Return an estimate of the cost by multiplying the count of unsatisfied goal facts
       by a fixed factor (in this case, 2). Our rationale is that each ring typically
       undergoes at least one or two rod-swapping actions to move into place, especially
       if it needs to be removed from somewhere before being placed correctly.
    4. If the state already satisfies all goal facts (i.e., it is a goal state), the
       heuristic cost is 0.

    This simple strategy remains computationally cheap and often yields a smaller number
    of expanded nodes compared to a purely blind search.
    """

    def __init__(self, task):
        """
        Extracts the goal conditions from the task and stores them for later use.

        Parameters
        ----------
        task : Task
            The planning task object, containing goals, operators, initial state, etc.
        """
        self.goals = task.goals
        # We do not need to parse static facts in this domain; they are often empty.
        # If needed, we could do so here:
        #   self.static = task.static
        #   (Potentially parse or store domain invariants.)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state, represented by 'node'.

        Parameters
        ----------
        node : SearchNode
            A node in the search tree whose state needs evaluation.

        Returns
        -------
        int
            Estimated cost (number of additional actions) to reach a goal.
        """
        state = node.state

        # Count how many goal facts are not satisfied in the current state.
        unsatisfied_goals = sum(1 for g in self.goals if g not in state)

        # Multiply by a small factor to account for the typical number of swaps.
        return 2 * unsatisfied_goals