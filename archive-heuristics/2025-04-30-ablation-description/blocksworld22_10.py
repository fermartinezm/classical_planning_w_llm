from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by counting the
    number of goal conditions that are not currently satisfied in the state.
    Each unsatisfied goal predicate is counted as 1.

    # Assumptions
    - The heuristic assumes that achieving each unsatisfied goal predicate
      requires at least one action. This is a relaxation of the problem where
      preconditions and interactions between actions are ignored.
    - The heuristic does not distinguish between different types of goal
      predicates (e.g., `on`, `on-table`, `clear`) or the complexity of
      achieving them.

    # Heuristic Initialization
    - The heuristic stores the set of goal predicates from the planning task.
    - The Blocksworld domain, as provided, does not contain static facts,
      so no static information needs to be processed or stored.

    # Step-By-Step Thinking for Computing Heuristic
    1. Retrieve the set of goal predicates defined in the planning task.
    2. Retrieve the set of facts that are true in the current state.
    3. Compute the set difference between the goal predicates and the current
       state facts. This results in a set containing all goal predicates
       that are *not* present in the current state.
    4. The heuristic value is the number of predicates in this set difference.
       This count represents the number of goal conditions that still need
       to be achieved.
    5. If the current state is the goal state, the set difference will be empty,
       and the heuristic value will be 0. Otherwise, the value will be greater
       than 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.

        Args:
            task: The planning task object containing goals and static facts.
        """
        # Store the set of goal predicates.
        self.goals = task.goals
        # The blocksworld domain provided has no static facts, so we don't need to process task.static.

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        The heuristic value is the number of goal predicates not satisfied
        in the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # The heuristic is the number of goal predicates that are not in the current state.
        # This is equivalent to the size of the set difference between the goals and the state.
        unsatisfied_goals = self.goals - state

        return len(unsatisfied_goals)

