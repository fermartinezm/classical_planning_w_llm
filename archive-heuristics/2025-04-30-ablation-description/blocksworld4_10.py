from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of goal conditions that are not satisfied
    in the current state. It counts the number of goal predicates (like `on`,
    `on-table`, and `clear`) that are specified in the goal state but are not
    present in the current state.

    # Assumptions:
    - The cost of each action is uniform (e.g., 1).
    - The goal state is defined by a set of positive predicates.

    # Heuristic Initialization
    - The heuristic stores the set of goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Retrieve the set of goal predicates stored during initialization (`self.goals`).
    2. Retrieve the current state predicates from the input node (`node.state`).
    3. Calculate the set difference between the goal predicates and the current state predicates (`self.goals - node.state`). This results in a set containing all goal predicates that are *not* present in the current state.
    4. The heuristic value is the number of predicates in this difference set (i.e., the size of the set calculated in step 3).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal conditions.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # The set of facts that must hold in goal states.
        # These are already parsed into a frozenset of strings by the Task class.
        self.goals = task.goals
        # Blocksworld domain typically has no static facts relevant to the goal state structure.
        # We don't need to process task.static for this heuristic.

    def __call__(self, node):
        """
        Estimate the minimum cost to reach the goal state from the current state.

        The heuristic value is the number of goal predicates not satisfied.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state # The current state as a frozenset of strings.

        # The heuristic is the number of goal predicates that are NOT in the current state.
        # This is simply the size of the set difference: self.goals - state
        # The set difference operation efficiently finds elements in self.goals that are not in state.
        unsatisfied_goals = self.goals - state

        # The heuristic value is the count of unsatisfied goal predicates.
        heuristic_value = len(unsatisfied_goals)

        return heuristic_value
