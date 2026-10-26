from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack all blocks in the correct order as specified by the goal.

    # Assumptions:
    - Blocks can be on the table, on top of another block, or being held by the arm.
    - The arm can hold only one block at a time.
    - Each block must be moved individually if it is not in the correct position.

    # Heuristic Initialization
    - Extract the goal conditions to determine the target position for each block.
    - Store the current state and goal state for comparison.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the target position for each block from the goal conditions.
    2. For each block, determine if it is in the correct position.
    3. For blocks not in the correct position, count how many blocks are above them that also need to be moved.
    4. Sum these counts to estimate the total number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """
            Check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(on b1 b2)").
            - `args`: The pattern to match (e.g., "on", "*", "*").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        goal_positions = {}
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                _, obj, target = goal[1:-1].split()
                goal_positions[obj] = target

        current_positions = {}
        for fact in state:
            if match(fact, "on", "*", "*"):
                _, obj, pos = fact[1:-1].split()
                current_positions[obj] = pos

        total_actions = 0

        for block in current_positions:
            if block not in goal_positions:
                total_actions += 1
                continue
            target = goal_positions[block]
            current = current_positions[block]
            if current != target:
                above_blocks = []
                for other_block in current_positions:
                    if other_block != block and current_positions[other_block] == block:
                        above_blocks.append(other_block)
                total_actions += 1 + len(above_blocks)

        return total_actions
