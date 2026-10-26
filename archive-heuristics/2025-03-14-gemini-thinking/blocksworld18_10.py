from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    For each block, it checks if its 'on', 'on-table', and 'clear' predicates match the goal state.
    The heuristic value is the count of blocks that are not correctly positioned according to the goal.
    This heuristic is admissible if we consider each misplaced block requires at least one action to correct its position. However, for greedy best-first search, admissibility is not required, and this heuristic aims to efficiently guide the search.

    # Assumptions:
    - The goal state is defined by a set of 'on', 'on-table', and 'clear' predicates.
    - The heuristic assumes that each unsatisfied goal condition for a block requires at least one action to be resolved.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal predicates related to block positions ('on', 'on-table', 'clear') from the task goals.
    3. For each block mentioned in the goal predicates, check its current state against the goal state:
       a. For each goal predicate 'on(block1, block2)', check if 'on(block1, block2)' exists in the current state. If not, increment the heuristic value.
       b. For each goal predicate 'on-table(block)', check if 'on-table(block)' exists in the current state. If not, increment the heuristic value.
       c. For each goal predicate 'clear(block)', check if 'clear(block)' exists in the current state. If not, increment the heuristic value.
    4. Return the total incremented heuristic value as the estimated cost to reach the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Estimate the number of misplaced blocks based on goal conditions."""
        state = node.state
        heuristic_value = 0

        goal_on_predicates = []
        goal_on_table_predicates = []
        goal_clear_predicates = []

        for goal in self.goals:
            if goal.startswith('(on '):
                goal_on_predicates.append(goal)
            elif goal.startswith('(on-table '):
                goal_on_table_predicates.append(goal)
            elif goal.startswith('(clear '):
                goal_clear_predicates.append(goal)

        for goal_predicate in goal_on_predicates:
            if goal_predicate not in state:
                heuristic_value += 1

        for goal_predicate in goal_on_table_predicates:
            if goal_predicate not in state:
                heuristic_value += 1

        for goal_predicate in goal_clear_predicates:
            if goal_predicate not in state:
                heuristic_value += 1

        return heuristic_value
