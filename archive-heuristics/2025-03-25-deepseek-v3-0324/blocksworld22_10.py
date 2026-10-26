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
    1. Counting blocks that are not in their correct position in the goal state
    2. Considering the dependencies between blocks (a block can't be placed correctly until its supporting block is in place)
    3. Accounting for the need to clear blocks that are in the way of correct placements

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - The goal state specifies exact positions for all blocks (complete goal specification)

    # Heuristic Initialization
    - Extract the goal conditions and build a mapping of each block to its required supporting block
    - Identify which blocks should be on the table in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: +1 (putdown action)
       - If on another block in goal but not in current state: +1 (stack action)
    2. For blocks that need to be moved:
       - If the block is currently under another block: +1 (unstack action to clear it)
       - If the block we need to stack onto isn't ready: +1 for each such dependency
    3. If the arm is holding a block that's not in its correct position: +1 (either putdown or stack)
    4. The total heuristic is the sum of all these required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: {block: (support_block or None if on table)}
        self.goal_structure = {}
        self.goal_on_table = set()
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, support = args
                self.goal_structure[block] = support
            elif predicate == "on-table":
                block = args[0]
                self.goal_structure[block] = None
                self.goal_on_table.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Track which blocks are currently being held
        holding = None
        for fact in state:
            if match(fact, "holding", "*"):
                holding = get_parts(fact)[1]
                break
        
        # Build current block structure
        current_structure = {}
        current_on_table = set()
        blocks_above = {}  # {block: set of blocks above it}
        
        for block in self.goal_structure:
            blocks_above[block] = set()
            
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, support = args
                current_structure[block] = support
                blocks_above[support].add(block)
            elif predicate == "on-table":
                block = args[0]
                current_structure[block] = None
                current_on_table.add(block)
        
        heuristic_value = 0
        
        # Check each block's position
        for block, goal_support in self.goal_structure.items():
            current_support = current_structure.get(block, None)
            
            # Block is in correct position
            if current_support == goal_support:
                continue
                
            # Block needs to be moved
            heuristic_value += 1
            
            # If it's currently under another block, need to unstack those first
            if blocks_above[block]:
                heuristic_value += len(blocks_above[block])
            
            # If we need to stack it on another block that's not ready
            if goal_support is not None and current_structure.get(goal_support, None) != self.goal_structure.get(goal_support, None):
                heuristic_value += 1
        
        # If holding a block that's not in its correct position
        if holding is not None:
            # Check if the held block is already in its correct position
            held_block_pos_ok = True
            for goal in self.goals:
                predicate, *args = get_parts(goal)
                if predicate == "holding" and args[0] == holding:
                    held_block_pos_ok = False
                    break
            
            if not held_block_pos_ok:
                heuristic_value += 1
        
        return heuristic_value
