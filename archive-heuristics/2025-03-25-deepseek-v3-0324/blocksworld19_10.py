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
    1. Counting blocks that are not in their correct position in the goal tower
    2. Considering whether the arm is empty or holding a block
    3. Accounting for necessary unstacking and restacking operations

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be placed on the table or on other blocks
    - The goal specifies a complete tower structure (no partial goals)

    # Heuristic Initialization
    - Extract the goal structure (tower relationships) from the task goals
    - Identify which blocks should be on the table in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the goal tower:
       - If not, it will need to be moved (either unstacked or picked up)
    2. For blocks that are in the wrong position:
       - If they're supporting other blocks, those need to be unstacked first
       - If they're below their correct position, they need to be moved aside
    3. Count the number of blocks not in their correct position
    4. Add additional costs for:
       - Blocks that need to be unstacked before they can be moved
       - The arm needing to be free to perform operations
       - Blocks that are currently being held
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: maps each block to what it should be on
        self.goal_on = {}
        self.goal_on_table = set()
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                self.goal_on_table.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Build current block relationships
        current_on = {}
        current_on_table = set()
        holding = None
        clear_blocks = set()
        
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under = args
                current_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                current_on_table.add(block)
            elif predicate == "holding":
                holding = args[0]
            elif predicate == "clear":
                clear_blocks.add(args[0])
        
        # Initialize heuristic value
        h = 0
        
        # Count blocks that are not in their correct position
        misplaced_blocks = set()
        
        # Check blocks that should be on table
        for block in self.goal_on_table:
            if block not in current_on_table:
                misplaced_blocks.add(block)
        
        # Check blocks that should be on other blocks
        for block, under in self.goal_on.items():
            if current_on.get(block) != under:
                misplaced_blocks.add(block)
                if under in current_on and current_on[under] == block:
                    # Blocks are in each other's positions - need to move both
                    misplaced_blocks.add(under)
        
        # For each misplaced block, add cost based on its current situation
        for block in misplaced_blocks:
            # If block is being held, we still need to place it
            if holding == block:
                h += 1
            else:
                # If block is under another block, need to unstack first
                if block in current_on.values():
                    h += 2  # unstack + move
                else:
                    h += 1  # simple move
                
                # If block is currently on something, need to pick it up
                if block in current_on:
                    h += 1
        
        # Additional cost if arm is not empty and we're not holding a misplaced block
        if holding is not None and holding not in misplaced_blocks:
            h += 1  # need to put down current block
        
        # Ensure at least minimal cost for non-goal states
        return max(1, h)
