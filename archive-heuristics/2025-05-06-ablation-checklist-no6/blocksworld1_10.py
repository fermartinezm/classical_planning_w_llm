from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the number of actions required to reach the goal
    by counting the number of goal conditions that are not currently satisfied
    in the state.

    It is a simple, non-admissible heuristic that provides a direct measure
    of how many goal predicates need to be made true.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the set of goal facts.

        Args:
            task: The planning task object, containing initial state, goals, etc.
        """
        # The set of facts that must hold in goal states.
        # These are already provided as a frozenset of strings in task.goals.
        self.goals = task.goals

        # Static facts are available in task.static, but the Blocksworld domain
        # provided does not define any static predicates or facts, so task.static
        # will be an empty frozenset. This heuristic does not require static facts.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic value is the number of goal facts that are not present
        in the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # The heuristic is the count of goal facts that are not true in the current state.
        # This is equivalent to the size of the set difference between the goal facts
        # and the current state facts.
        h_value = 0
        for goal_fact in self.goals:
            if goal_fact not in state:
                h_value += 1

        # The heuristic is 0 if and only if all goal facts are in the state,
        # which means the state is a goal state.
        # The heuristic value is finite for any state as the number of goal facts is finite.

        return h_value

