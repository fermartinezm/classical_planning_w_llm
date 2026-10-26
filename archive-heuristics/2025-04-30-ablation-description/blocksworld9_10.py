class blocksworldHeuristic:
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by counting the number
    of goal facts that are not currently true in the state.

    # Assumptions
    - The goal is specified as a conjunction of facts about block positions (on, on-table)
      and clear conditions.
    - The heuristic assumes that achieving each unsatisfied goal fact requires at least one action,
      and potentially more due to dependencies or necessary intermediate steps (like clearing blocks).
      By counting all unsatisfied goal facts, it provides a simple measure of distance.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state facts from the node.
    2. Initialize a counter for unsatisfied goal facts to 0.
    3. Iterate through each fact string in the set of goal facts stored during initialization.
    4. For each goal fact string, check if that exact string is present in the current state facts (which are also stored as strings).
    5. If a goal fact string is NOT present in the current state facts, increment the counter.
    6. The heuristic value is the final count of unsatisfied goal facts.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal facts."""
        # Store the set of goal facts.
        self.goal_facts = task.goals
        # Static facts are not needed for this heuristic.
        # static_facts = task.static

    def __call__(self, node):
        """Compute the number of goal facts not satisfied in the current state."""
        state = node.state

        # The heuristic is the number of goal facts that are not in the current state.
        unsatisfied_goals_count = 0
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                unsatisfied_goals_count += 1

        return unsatisfied_goals_count
