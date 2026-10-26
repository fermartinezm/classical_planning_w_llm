from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to stack all blocks in their target positions as specified in the goal state.

    # Assumptions:
    - Blocks can be moved individually and stacked on top of each other.
    - The goal state specifies a target configuration of blocks on tables or stacked on other blocks.
    - The arm can hold one block at a time and must be empty to move blocks.

    # Heuristic Initialization
    - Extracts the goal conditions to determine the target configuration of blocks.
    - Constructs a tree representation of the target configuration for efficient analysis.

    # Step-by-Step Thinking for Computing Heuristic
    1. **Check for Goal State**: If the current state matches the goal, return 0.
    2. **Extract Goal Configuration**: Parse the goal state to build a tree of target positions.
    3. **Analyze Current State**: Determine the current position and status (held or on a surface) of each block.
    4. **Calculate Required Actions**:
       - For each block not in its target position, calculate the number of moves required.
       - Consider blocks that need to be moved out of the way to access the target position.
       - Account for the arm's state (empty or holding a block) and its impact on the number of actions.
    5. **Sum Actions**: Aggregate the actions needed for all blocks to reach their target positions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract target positions for each block from the goal state
        self.goal_positions = {}
        self.goal_holding = None
        for fact in self.goals:
            parts = fact[1:-1].split()
            if parts[0] == 'on':
                obj, under = parts[1], parts[2]
                self.goal_positions[obj] = under
            elif parts[0] == 'on-table':
                obj = parts[1]
                self.goal_positions[obj] = 'table'
            elif parts[0] == 'holding':
                self.goal_holding = parts[1]

        # Build a tree structure for the goal configuration
        self.goal_tree = {}
        for block, under in self.goal_positions.items():
            if under == 'table':
                self.goal_tree[block] = None
            else:
                self.goal_tree[block] = under

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Check if current state is the goal state
        if self.goals.issubset(state):
            return 0

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        # Extract current positions and holding status
        current_positions = {}
        holding = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                obj, under = parts[1], parts[2]
                current_positions[obj] = under
            elif parts[0] == 'on-table':
                obj = parts[1]
                current_positions[obj] = 'table'
            elif parts[0] == 'holding':
                holding = parts[1]

        total_actions = 0

        # Process each block in the goal configuration
        for block in self.goal_positions:
            goal_pos = self.goal_positions[block]
            current_pos = current_positions.get(block, None)

            if current_pos == goal_pos:
                continue  # Block is already in correct position

            # If block is being held, it needs to be put down
            if holding == block:
                total_actions += 1  # Putdown action
                holding = None

            # If block is on another block, that block must be moved first
            if current_pos in current_positions and current_pos != 'table':
                under_block = current_pos
                while under_block in current_positions and current_positions[under_block] != 'table':
                    under_block = current_positions[under_block]
                if under_block != goal_pos:
                    total_actions += 1  # Move the under_block

            # Move the block to its target position
            if goal_pos != 'table':
                # If target is another block, ensure it's in place
                if self.goal_tree[block] is not None:
                    target_under = self.goal_tree[block]
                    if current_positions.get(target_under, None) != goal_pos:
                        total_actions += 1  # Stack the block
            else:
                total_actions += 1  # Putdown on table

            # If arm is empty after putting down, it may need to pick up again
            if goal_pos != 'table' and holding is None:
                total_actions += 1  # Pickup action

        return total_actions
