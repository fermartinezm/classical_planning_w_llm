from heuristics.heuristic_base import Heuristic
# No need for fnmatch as we parse facts manually

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like '(predicate arg1 arg2)'
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either misplaced
    relative to their goal position or are blocking a block that needs to be clear,
    plus any blocks stacked on top of them. It also adds a penalty if the arm is not empty.
    The heuristic counts the number of blocks that need to be moved (picked up or unstacked)
    at least once to resolve these issues.

    # Assumptions:
    - The goal specifies a desired configuration of blocks using 'on', 'on-table', and 'clear' predicates.
    - Blocks not mentioned in 'on' or 'on-table' goal predicates do not have a specific goal location relative to other blocks, but might need to be clear.
    - The arm must be empty to pick up a block or unstack.

    # Heuristic Initialization
    - Extracts the goal structure: which block should be on which other block or the table (`goal_parent_map`).
    - Identifies which blocks must be clear in the goal state (`goal_clear_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize cost to 0.
    2. Identify blocks that are "problematic":
       a. Blocks that are part of a goal stack (appear as keys in `goal_parent_map`) but are currently not on their correct goal parent (or table).
       b. Blocks that are required to be clear in the goal (`goal_clear_blocks`) but are currently not clear (something is on top of them).
    3. Create a set `blocks_to_move` and add all "problematic" blocks to it. These blocks need to be moved at least once.
    4. For each block identified in step 2 (problematic blocks), traverse the stack upwards from it in the current state. Add every block found directly on top to the `blocks_to_move` set. These blocks must be moved *before* the problematic block underneath can be accessed or moved. Repeat this for blocks found further up the stack.
    5. The heuristic value is the total number of unique blocks in the `blocks_to_move` set.
    6. Add 1 to the cost if the robot's arm is not empty, as the held block usually needs to be put down or stacked before other operations can occur, potentially adding an action.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and clear requirements.
        """
        self.goals = task.goals

        # Build goal_parent_map: block -> block_it_should_be_on or 'table'
        self.goal_parent_map = {}
        # Identify blocks that must be clear in the goal
        self.goal_clear_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, parent = parts[1], parts[2]
                self.goal_parent_map[block] = parent
            elif predicate == "on-table":
                block = parts[1]
                self.goal_parent_map[block] = 'table'
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear_blocks.add(block)
            # Ignore arm-empty goal if present, it's handled as a state penalty

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify blocks that are in the wrong place relative to their goal parent.
        misplaced_relative_to_parent = set()
        for block, goal_parent in self.goal_parent_map.items():
            current_parent = None
            # Check if block is on the table
            if f'(on-table {block})' in state:
                current_parent = 'table'
            else:
                # Check if block is on another block
                for fact in state:
                    # Fact must be '(on block ?y)'
                    if get_parts(fact)[0] == "on" and get_parts(fact)[1] == block:
                        current_parent = get_parts(fact)[2]
                        break # Found the block below 'block'

            # If the block is supposed to be somewhere (on another block or table)
            # and it's not in the correct place relative to its goal parent
            if current_parent != goal_parent:
                misplaced_relative_to_parent.add(block)

        # Identify blocks that are supposed to be clear but are not.
        not_clear_when_should_be = set()
        for block in self.goal_clear_blocks:
            if f'(clear {block})' not in state:
                not_clear_when_should_be.add(block)

        # The set of blocks that are "problematic" (misplaced or not clear when they should be).
        problematic_blocks = misplaced_relative_to_parent.union(not_clear_when_should_be)

        # The set of blocks that need to be moved (picked up/unstacked).
        # This includes problematic blocks and anything stacked on top of them.
        blocks_to_move = set()

        # Add the problematic blocks themselves to the set of blocks that need moving.
        blocks_to_move.update(problematic_blocks)

        # Add blocks stacked on top of problematic blocks.
        for block_under in problematic_blocks:
            current = block_under
            while True:
                block_on_top = None
                # Find the block directly on top of 'current'
                for fact in state:
                    # Fact must be '(on ?x current)'
                    if get_parts(fact)[0] == "on" and get_parts(fact)[2] == current:
                        block_on_top = get_parts(fact)[1]
                        break
                if block_on_top:
                    blocks_to_move.add(block_on_top)
                    current = block_on_top
                else:
                    break # Reached the top of the current stack

        # The heuristic value is the count of unique blocks that need to be moved.
        cost = len(blocks_to_move)

        # Add 1 if the arm is holding a block, as it might need to be put down.
        if '(arm-empty)' not in state:
            cost += 1

        return cost
