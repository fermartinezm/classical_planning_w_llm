from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This is a domain-dependent heuristic for the 'rod-rings' domain. It estimates
    how many actions are needed to bring the current state closer to satisfying
    all goal facts. Although this heuristic is not necessarily admissible, it is
    designed to be computationally fast and to reduce the number of expanded nodes
    in a greedy best-first search.

    Assumptions
    -----------
    1. The domain requires achieving specific predicates among:
       - (on ?ring1 ?ring2)
       - (top ?ring ?stick)
       - (bot ?ring ?stick)
       - (holding ?ring)
    2. A goal state is any state in which all goal facts (as specified by the PDDL
       problem file) are satisfied.
    3. No static facts in this domain affect the cost calculation directly (i.e.,
       we can ignore them or store them for potential use).

    Heuristic Initialization
    ------------------------
    - We store the goal facts in the constructor. Since there are no relevant static
      facts (empty frozenset for this domain), we do not use them here.
    - The heuristic is 0 whenever the goal subset is contained in the current state.
    - Otherwise, the heuristic is computed according to how many goal facts remain
      unsatisfied.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the goal is already achieved (i.e., if all goal facts are included in
       the current state). If yes, return 0.
    2. Otherwise, compute how many goal facts are not yet satisfied by the current state.
    3. Estimate that each unsatisfied goal fact will require a small number of actions
       to fix, so multiply the number of unsatisfied goal facts by a constant factor
       (e.g. 2).
    4. Return this product as the heuristic value.

    This simple domain-dependent approach uses the fact that 'holding', 'top', 'bot',
    and 'on' are the only predicates relevant to rod-rings. By directly counting which
    of these facts in the goal are not in the current state, we get a fast but
    reasonably informed heuristic.
    """

    def __init__(self, task):
        """
        Store goal conditions and (optionally) any static facts if needed.
        
        Parameters
        ----------
        task : Task
            The planner's internal task representation, which holds goals,
            initial state, and static facts.
        """
        self.goals = task.goals
        self.static = task.static  # Not used in this simple heuristic, but available.

    def __call__(self, node):
        """
        Calculate the heuristic value for the given search node.

        Parameters
        ----------
        node : Node
            The search node containing the current state to be evaluated.

        Returns
        -------
        int
            A non-admissible estimate of the number of actions needed to
            reach the goal state. Returns 0 if all goals are satisfied.
        """
        state = node.state

        # If all goals are satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Count the number of goal facts not in the current state
        unsatisfied_goals = self.goals - state

        # Multiply by a factor (e.g. 2) to reflect that moving or reordering a ring
        # typically involves multiple steps.
        return 2 * len(unsatisfied_goals)