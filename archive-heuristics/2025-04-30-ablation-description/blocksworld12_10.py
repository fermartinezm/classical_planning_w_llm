class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by counting the
    number of goal conditions that are not satisfied in the current state.
    It provides a simple measure of how many desired facts are missing.

    # Assumptions:
    - The goal is defined as a conjunction of positive ground literals
      (predicates like `on`, `on-table`, `clear`, `arm-empty`).
    - The heuristic counts how many of these specific goal facts are false
      in the current state.
    - This heuristic is not admissible but can be effective for greedy best-first search.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task definition.
    - Static facts are not used by this heuristic as blocksworld has no static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the set of goal facts (`self.goals`) stored during initialization.
    2. Access the set of facts true in the current state (`node.state`).
    3. Compute the set difference between the goal facts and the current state facts (`self.goals - state`). This set contains all goal facts that are *not* present in the current state.
    4. The heuristic value is the number of facts in this set difference, i.e., the count of unsatisfied goal conditions.
    5. Return this count.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not needed for this heuristic as blocksworld has no static facts.
        # self.static = task.static # Blocksworld static facts are empty

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state."""
        state = node.state

        # The heuristic is the number of goal facts not present in the current state.
        # This is equivalent to the size of the set difference: goals - state.
        # The size of the set difference is the number of elements in 'goals'
        # that are NOT in 'state'.
        unsatisfied_goals = self.goals - state

        # The heuristic value is the count of unsatisfied goal facts.
        heuristic_value = len(unsatisfied_goals)

        return heuristic_value
