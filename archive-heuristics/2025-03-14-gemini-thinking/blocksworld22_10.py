from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their goal positions.
    For each block, it checks if its 'on' or 'on-table' predicate matches the goal state.
    If a block is not in its goal position, it increments the heuristic value by 1.
    This heuristic is admissible in relaxed planning sense, as it counts the minimum number of blocks that need to be moved.

    # Assumptions:
    - The goal state is defined by a set of 'on' and 'on-table' predicates.
    - All blocks mentioned in the goal configuration are considered for the heuristic calculation.

    # Heuristic Initialization
    - Extract the goal block positions from the task's goal predicates.
    - Store the goal positions in a dictionary for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the goal 'on' and 'on-table' predicates from the task definition.
    3. Create a dictionary to store the goal position for each block.
       - For each goal predicate 'on(block1, block2)', store block1's goal position as block2.
       - For each goal predicate 'on-table(block)', store block's goal position as 'table'.
    4. Iterate through the blocks that are part of the goal configuration (keys of the goal positions dictionary).
    5. For each block, determine its current position from the current state.
       - Check for 'on(block, block_under)' predicate in the current state. If found, current position is block_under.
       - If 'on' predicate is not found, check for 'on-table(block)' predicate. If found, current position is 'table'.
       - If neither 'on' nor 'on-table' is found (which should not happen in well-formed blocksworld problems given the preconditions of actions), assume it's not in a goal position.
    6. Compare the current position with the goal position for the block.
    7. If the current position is different from the goal position, increment the heuristic value by 1.
    8. Return the total heuristic value, which represents the estimated number of misplaced blocks.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal block positions."""
        self.goals = task.goals
        self.goal_block_positions = {}

        for goal_fact in self.goals:
            parts = goal_fact[1:-1].split()
            predicate = parts[0]
            if predicate == 'on':
                block1, block2 = parts[1], parts[2]
                self.goal_block_positions[block1] = block2
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_block_positions[block] = 'table'

    def __call__(self, node):
        """Estimate the number of misplaced blocks in the current state."""
        state = node.state
        misplaced_blocks = 0

        for block, goal_position in self.goal_block_positions.items():
            current_position = None
            for fact in state:
                parts = fact[1:-1].split()
                predicate = parts[0]
                if predicate == 'on' and parts[1] == block:
                    current_position = parts[2]
                    break
                elif predicate == 'on-table' and parts[1] == block:
                    current_position = 'table'
                    break

            if current_position != goal_position:
                misplaced_blocks += 1

        return misplaced_blocks
