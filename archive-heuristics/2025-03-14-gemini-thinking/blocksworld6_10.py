from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by counting the number of blocks that are not in their goal positions and stacks, and the number of blocks that are not clear when they should be in the goal.

    # Assumptions:
    - The goal state is defined by a set of `on`, `on-table`, and `clear` predicates.
    - The heuristic assumes that for each misplaced block, at least one action is required to move it to its correct position.
    - It does not consider the optimal sequence of actions, but rather focuses on the number of blocks that are out of place.

    # Heuristic Initialization
    - Extract the goal predicates (`on`, `on-table`, `clear`) from the task definition.
    - Store the goal `on` relations in a dictionary for efficient lookup, mapping each block to the block it should be on.
    - Store the goal `on-table` blocks in a set.
    - Store the goal `clear` blocks in a set.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Parse the goal state and store the goal `on`, `on-table`, and `clear` predicates.
    3. For each block mentioned in a goal `on` or `on-table` predicate:
        a. Determine the block it should be on in the goal state (or 'table' if it should be on the table).
        b. Determine the block it is currently on in the current state (or 'table' if it is on the table).
        c. If the current 'on' relation does not match the goal 'on' relation, increment the heuristic value.
    4. For each block that should be `clear` in the goal state:
        a. Check if the `clear` predicate holds for this block in the current state.
        b. If the `clear` predicate does not hold, increment the heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the blocksworld heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()

        for goal in self.goals:
            parts = self._get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block1, block2 = parts[1], parts[2]
                self.goal_on[block1] = block2
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_on_table.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        current_on = {}
        current_on_table = set()
        current_clear = set()

        for fact in state:
            parts = self._get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block1, block2 = parts[1], parts[2]
                current_on[block1] = block2
            elif predicate == 'on-table':
                block = parts[1]
                current_on_table.add(block)
            elif predicate == 'clear':
                block = parts[1]
                current_clear.add(block)

        for block in self.goal_on:
            goal_under_block = self.goal_on[block]
            current_under_block = current_on.get(block, 'table') if block in current_on else 'table' if block in current_on_table else None

            if goal_under_block != current_under_block:
                heuristic_value += 1

        for block in self.goal_on_table:
            if block not in current_on_table:
                heuristic_value += 1

        for block in self.goal_clear:
            if block not in current_clear:
                heuristic_value += 1

        return heuristic_value

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()
