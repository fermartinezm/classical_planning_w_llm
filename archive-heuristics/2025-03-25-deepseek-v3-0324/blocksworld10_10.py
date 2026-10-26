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
    2. Considering the need to move blocks to clear others
    3. Accounting for the arm state (whether it's empty or holding a block)
    4. Adding penalties for blocks that need to be moved multiple times

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - Blocks must be clear to be moved

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block configuration
    - Build a mapping of which block should be on which other block (or table)
    - Identify the top-most blocks in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: +1 (putdown needed)
       - If on another block in goal but not in current state: +2 (unstack and stack needed)
    2. For blocks that are in the wrong position:
       - If the block is currently supporting other blocks: +1 for each block above it
    3. If the arm is holding a block that's not in its goal position: +1
    4. If the arm is empty but needs to pick up a block: +1
    5. If a block is preventing another block from being moved: +1
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: {block: (on-table/on, target)}
        self.goal_structure = {}
        self.goal_clear = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_structure[parts[1]] = ("on", parts[2])
            elif parts[0] == "on-table":
                self.goal_structure[parts[1]] = ("on-table", None)
            elif parts[0] == "clear":
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Build current block structure
        current_on = {}  # block: what it's on
        current_above = {}  # block: set of blocks above it
        current_clear = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
                if parts[2] not in current_above:
                    current_above[parts[2]] = set()
                current_above[parts[2]].add(parts[1])
            elif parts[0] == "on-table":
                current_on[parts[1]] = None
            elif parts[0] == "clear":
                current_clear.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
            elif parts[0] == "arm-empty":
                arm_empty = True
        
        h = 0
        
        # Check each block's position
        for block in self.goal_structure:
            goal_type, goal_target = self.goal_structure[block]
            current_target = current_on.get(block, None)
            
            # Block is in correct position
            if (goal_type == "on-table" and current_target is None) or \
               (goal_type == "on" and current_target == goal_target):
                continue
                
            # Block needs to be moved
            h += 2  # pickup + putdown/stack
            
            # If block is under other blocks, need to move them first
            if block in current_above:
                h += len(current_above[block])
                
            # If target position is not clear in current state
            if goal_type == "on" and goal_target not in current_clear:
                h += 1
        
        # Handle arm state
        if holding:
            # If holding a block that's not in its goal position
            if holding in self.goal_structure:
                h += 1  # need to put it down
        elif not arm_empty:
            h += 1  # need to release the arm
            
        return h
