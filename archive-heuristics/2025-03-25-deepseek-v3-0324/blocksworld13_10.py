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
    2. Considering the dependencies between blocks (a block must be moved before its supporting block)
    3. Accounting for the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The arm can hold only one block at a time.
    - Blocks can only be stacked one on top of another.
    - The table has unlimited space for placing blocks.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block configuration.
    - Build a mapping of which blocks should be on which other blocks in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on-table in goal but not in current state, it needs to be moved.
       - If on another block in goal but not in current state, it needs to be moved.
    2. For blocks that are in the wrong position:
       - If the block is clear, it can be moved directly (1 action to pick up).
       - If the block is under another block, we need to unstack blocks above it first.
    3. For blocks that are correct but have incorrect blocks above them:
       - These will be handled by moving the incorrect blocks above them.
    4. The total heuristic is the sum of:
       - 1 for each block that needs to be moved (pickup/putdown or stack/unstack)
       - 1 for each block that needs to be cleared before moving the target block
       - 1 if the arm is currently holding a block (we need to put it down first)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: what should be on what
        self.goal_on = {}  # block -> what it should be on (None for table)
        self.goal_under = {}  # block -> what should be on top of it
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                self.goal_on[block] = under
                self.goal_under[under] = block
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_on[block] = None

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Check if arm is holding a block
        holding = None
        for fact in state:
            if match(fact, "holding", "*"):
                holding = get_parts(fact)[1]
                break
        
        # Build current block structure
        current_on = {}  # block -> what it's currently on
        current_under = {}  # block -> what's currently on top of it
        clear_blocks = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
                current_under[under] = block
            elif parts[0] == "on-table":
                block = parts[1]
                current_on[block] = None
            elif parts[0] == "clear":
                block = parts[1]
                clear_blocks.add(block)
        
        # Calculate heuristic value
        h = 0
        
        # If holding a block, we need at least 1 action to put it down
        if holding is not None:
            h += 1
        
        # Check each block's position
        for block in self.goal_on:
            goal_under = self.goal_on[block]
            current_under_block = current_on.get(block, None)
            
            # Block is in wrong position
            if goal_under != current_under_block:
                h += 1  # at least one action to move it
                
                # If it's not clear, need to unstack blocks above it first
                if block not in clear_blocks:
                    h += 1
        
        return h
