from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    For each block, it checks if it is on the correct block or on the table as specified in the goal.
    If a block is not in its goal position, it increments the heuristic value.

    # Assumptions:
    - The goal state is defined by a set of `on` and `on-table` predicates.
    - We are only concerned with the vertical arrangement of blocks.
    - The heuristic is not necessarily admissible but aims to be informative and efficiently computable.

    # Heuristic Initialization
    - Extract the goal block configuration from the task's goal predicates.
    - Store the goal parent for each block. If a block should be on the table, the goal parent is 'table'.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Parse the goal predicates to determine the desired parent for each block.
       Store this information in a dictionary `goal_block_parents` where keys are blocks and values are their goal parents (block name or 'table').
    3. Parse the current state to determine the current parent for each block.
       Store this information in a dictionary `current_block_parents` where keys are blocks and values are their current parents (block name or 'table').
    4. Iterate through all blocks mentioned in the goal configuration (keys of `goal_block_parents`).
    5. For each block, compare its `goal_parent` with its `current_parent`.
    6. If the `goal_parent` and `current_parent` are different, increment the heuristic value by 1.
    7. Return the total heuristic value.

    This heuristic essentially counts the number of blocks that are not in their correct vertical position according to the goal.
    It assumes that for each misplaced block, at least one action is required to move it to its correct position.
    """

    def __init__(self, task):
        """Initialize the blocksworld heuristic by extracting goal block configurations."""
        self.goals = task.goals
        self.static_facts = task.static
        self.goal_block_parents = {}
        self.blocks_in_goal = set()

        for goal_fact in self.goals:
            parts = goal_fact[1:-1].split()
            predicate = parts[0]
            if predicate == 'on':
                block, parent = parts[1], parts[2]
                self.goal_block_parents[block] = parent
                self.blocks_in_goal.add(block)
                self.blocks_in_goal.add(parent)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_block_parents[block] = 'table'
                self.blocks_in_goal.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.blocks_in_goal.add(block)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_block_parents = {}

        for fact in state:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == 'on':
                block, parent = parts[1], parts[2]
                current_block_parents[block] = parent
            elif predicate == 'on-table':
                block = parts[1]
                current_block_parents[block] = 'table'
            elif predicate == 'holding':
                block = parts[1]
                current_block_parents[block] = 'arm'


        heuristic_value = 0
        for block in self.blocks_in_goal:
            goal_parent = self.goal_block_parents.get(block, None)
            current_parent = current_block_parents.get(block, None)

            if goal_parent is not None:
                if current_parent != goal_parent:
                    heuristic_value += 1

        # Ensure heuristic is 0 for goal states
        if task.goal_reached(node.state, self.goals):
            return 0

        return heuristic_value

from heuristics.heuristic_base import Heuristic # required to avoid import error in test environment
from fnmatch import fnmatch # required to avoid import error in test environment
