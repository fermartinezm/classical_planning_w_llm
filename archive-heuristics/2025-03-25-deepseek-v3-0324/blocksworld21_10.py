from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by:
    1. Counting mismatched blocks (blocks not in their goal position)
    2. Considering the need to clear blocks below mismatched blocks
    3. Accounting for the arm state (whether it's empty or holding a block)
    4. Adding penalties for blocks that need to be moved multiple times

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - The goal is to achieve a specific stacking configuration

    # Heuristic Initialization
    - Extract the goal configuration (which block is on which other block or table)
    - Identify all blocks in the problem
    - Store static information (though blocksworld has no static facts)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the current state
    2. For blocks not in correct position:
       - If it's on the wrong block or table, count 1 action to move it
       - If the block it should be on is not clear, count 1 action to clear it
    3. If the arm is holding a block that's not in its correct position, count 1 action to put it down
    4. For each block that needs to be moved and is currently supporting other blocks, 
       count additional actions to unstack those blocks first
    5. The total heuristic is the sum of all these required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and identifying all blocks."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract all blocks from the initial state
        self.blocks = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                self.blocks.update(parts[1:])
        
        # Build goal configuration: maps each block to what it should be on
        self.goal_config = {}
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                _, block, under = get_parts(goal)
                self.goal_config[block] = under
            elif match(goal, "on-table", "*"):
                _, block = get_parts(goal)
                self.goal_config[block] = 'table'
            elif match(goal, "clear", "*"):
                _, block = get_parts(goal)
                # Clear goals are handled by the on/on-table goals
                pass

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
        
        # Build current configuration and what blocks are clear
        current_config = {}
        clear_blocks = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "on", "*", "*"):
                _, block, under = parts
                current_config[block] = under
            elif match(fact, "on-table", "*"):
                _, block = parts
                current_config[block] = 'table'
            elif match(fact, "clear", "*"):
                _, block = parts
                clear_blocks.add(block)
            elif match(fact, "holding", "*"):
                _, block = parts
                holding = block
            elif match(fact, "arm-empty"):
                arm_empty = True
        
        # If holding a block that's not in correct position, need to put it down first
        if holding:
            if holding in self.goal_config:
                h += 1  # Need to put it down before doing anything else
        
        # For each block, check if it's in correct position
        for block in self.blocks:
            if block not in self.goal_config:
                continue  # This block isn't mentioned in goals
                
            goal_under = self.goal_config[block]
            current_under = current_config.get(block, None)
            
            # Block is in wrong position
            if current_under != goal_under:
                h += 1  # At least one move to get it to right place
                
                # If it's on wrong block, that block might need to be cleared
                if current_under != 'table' and current_under in self.goal_config:
                    h += 1  # Need to move the block it's currently on
                
                # If it should be on another block, that block might not be clear
                if goal_under != 'table' and goal_under not in clear_blocks:
                    h += 1  # Need to clear the target block
        
        # Additional penalty for blocks that are supporting others but need to be moved
        supporting_blocks = set(current_config.values()) - {'table'}
        for block in supporting_blocks:
            if block in self.goal_config and current_config.get(block, None) != self.goal_config[block]:
                h += 1  # Need to unstack blocks above this one first
        
        return h
