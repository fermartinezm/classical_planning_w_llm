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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)


class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state by:
    1. Counting blocks that are not in their correct position in the goal state
    2. Considering whether the arm is empty or holding a block
    3. Accounting for necessary unstacking and stacking operations

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space for placing blocks

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block arrangement
    - Build a mapping of which blocks should be on which other blocks or on the table

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the current state:
       - If on the correct block (or table) and supporting the correct blocks, no action needed
       - Otherwise, it will need to be moved
    2. For blocks that need to be moved:
       - If the block is buried under others, it will need unstack operations for each block above it
       - If the block needs to be placed under another block, that block may need to be moved first
    3. The heuristic value is the sum of:
       - 1 for each block not in its correct position (pickup/putdown or stack/unstack)
       - 1 for each block that needs to be unstacked from above it
       - 1 if the arm is currently holding a block that's not in its goal position
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: what should be on what
        self.goal_on = {}  # block -> what it should be on (None for table)
        self.goal_clear = set()  # blocks that should be clear in goal
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                self.goal_on[parts[1]] = None
            elif parts[0] == "clear":
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Build current block structure
        current_on = {}  # block -> what it's currently on (None for table)
        current_above = {}  # block -> what's directly above it
        current_clear = set()  # blocks that are currently clear
        
        holding = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
                current_above[parts[2]] = parts[1]
            elif parts[0] == "on-table":
                current_on[parts[1]] = None
            elif parts[0] == "clear":
                current_clear.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
        
        # Calculate heuristic value
        h = 0
        
        # If holding a block that's not in its goal position, need to put it down first
        if holding is not None:
            h += 1
            
        # For each block, check if it's in the correct position
        for block in self.goal_on:
            # Check if block is on the correct thing
            goal_target = self.goal_on.get(block)
            current_target = current_on.get(block)
            
            if goal_target != current_target:
                h += 1  # Need to move this block
                
                # If it's buried under other blocks, need to unstack them first
                if block in current_above:
                    above = current_above[block]
                    while above in current_above:
                        h += 1  # Each unstack operation
                        above = current_above[above]
                    h += 1  # For the topmost block
                    
            # Check if the block should be clear and isn't
            if block in self.goal_clear and block not in current_clear:
                h += 1  # Need to clear this block
                
        return h
