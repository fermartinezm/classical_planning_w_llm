# from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of goal facts that are not currently true in the state.

    # Assumptions
    - The heuristic assumes a standard Blocksworld domain where the goal is
      defined by a set of facts specifying the desired arrangement of blocks
      (e.g., which block is on which, which blocks are on the table, and which
      blocks are clear).
    - The heuristic counts only the unsatisfied goal facts. It does not
      consider the dependencies between facts or the actions required to achieve
      them.

    # Heuristic Initialization
    - The heuristic stores the set of goal facts from the planning task.
    - Static facts are not used by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    1. Retrieve the set of goal facts stored during initialization.
    2. Retrieve the current state facts from the input node.
    3. Initialize a counter for unsatisfied goal facts to zero.
    4. Iterate through each fact in the set of goal facts.
    5. For each goal fact, check if it is present in the current state facts.
    6. If a goal fact is not found in the current state, increment the counter.
    7. After checking all goal facts, the value of the counter is the heuristic estimate.
    8. Return the counter value.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal facts."""
        # Store the set of goal facts.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost by counting unsatisfied goal facts."""
        state = node.state  # Current world state as a frozenset of fact strings.

        unsatisfied_goals_count = 0
        for goal_fact in self.goals:
            # Check if the goal fact is present in the current state.
            if goal_fact not in state:
                unsatisfied_goals_count += 1

        return unsatisfied_goals_count
