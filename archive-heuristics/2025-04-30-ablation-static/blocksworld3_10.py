from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the number of goal conditions related to block positions (on or on-table) that are not currently satisfied. Each unsatisfied position goal contributes 1 to the heuristic value.

    # Assumptions
    - The goal state is defined primarily by the desired configuration of blocks in stacks or on the table.
    - Goal conditions involving 'clear' or 'arm-empty' are typically satisfied implicitly when the block positions are correct, or are less critical for estimating the *main* work (moving blocks). This heuristic focuses only on 'on' and 'on-table' goal facts.
    - The cost of achieving each unsatisfied position goal is assumed to be at least 1, independent of other goals. This is a relaxation.

    # Heuristic Initialization
    - Extracts the set of goal facts from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter for unsatisfied position goal facts to 0.
    2. Iterate through each fact string in the set of goal facts (`task.goals`).
    3. Check if the goal fact string represents a block's position, i.e., if it starts with '(on ' or '(on-table '.
    4. If the goal fact is a position fact, check if this exact string exists in the current state facts (`node.state`).
    5. If the position goal fact is *not* found in the current state, increment the counter.
    6. The final value of the counter is the heuristic estimate for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the goal facts.
        """
        # Store the set of goal facts.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # self.static = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic value is the count of 'on' and 'on-table' goal facts
        that are not present in the current state.
        """
        state = node.state # Current world state facts (frozenset of strings)

        unsatisfied_position_goals = 0

        # Iterate through all goal facts
        for goal_fact in self.goals:
            # Check if the goal fact is about block position ('on' or 'on-table')
            # We check the start of the string after the opening parenthesis
            # This is a simple way to check the predicate name without full parsing
            if goal_fact.startswith('(on ') or goal_fact.startswith('(on-table '):
                # Check if this specific goal fact is present in the current state
                if goal_fact not in state:
                    unsatisfied_position_goals += 1

        return unsatisfied_position_goals
