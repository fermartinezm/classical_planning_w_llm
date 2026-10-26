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
    2. Considering the dependencies between blocks (a block can't be moved until blocks above it are moved)
    3. Accounting for the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The arm can hold only one block at a time.
    - Blocks can only be stacked on other blocks or on the table.
    - The goal specifies a complete configuration (all blocks must be in their specified positions).

    # Heuristic Initialization
    - Extract the goal conditions and build data structures to represent the goal configuration.
    - Identify all blocks in the domain.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the current state:
       - If on-table in goal but not in current state, or vice versa → mismatch
       - If on another block in goal but not the same block in current state → mismatch
    2. For each mismatched block:
       - If the block is clear (can be moved directly), count 1 action (pick/stack/unstack)
       - If the block is under other blocks, count actions needed to clear it (unstack all blocks above it)
    3. If the arm is holding a block, count 1 action to put it down or stack it
    4. Additional actions may be needed to:
       - Move blocks that are in the way of building the goal stack
       - Temporarily place blocks on the table to rearrange the stack
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal configuration
        self.goal_on = {}  # block -> what it's on (None if on table)
        self.goal_on_table = set()  # blocks that should be on table
        self.goal_clear = set()  # blocks that should be clear
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                self.goal_on_table.add(block)
            elif predicate == "clear":
                block = args[0]
                self.goal_clear.add(block)
        
        # Identify all blocks in the domain
        self.blocks = set()
        for fact in task.facts:
            parts = get_parts(fact)
            if parts[0] in ["on", "on-table", "clear", "holding"]:
                self.blocks.update(parts[1:])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Build current configuration
        current_on = {}  # block -> what it's on (None if on table)
        current_on_table = set()  # blocks currently on table
        current_clear = set()  # blocks currently clear
        holding = None  # block currently being held
        
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under = args
                current_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                current_on_table.add(block)
            elif predicate == "clear":
                block = args[0]
                current_clear.add(block)
            elif predicate == "holding":
                block = args[0]
                holding = block
        
        # Calculate heuristic value
        h = 0
        
        # 1. Count mismatched blocks
        for block in self.blocks:
            # Check if block is on correct surface
            goal_surface = self.goal_on.get(block, None)
            current_surface = current_on.get(block, None)
            
            if (goal_surface is None and block not in self.goal_on_table) or \
               (current_surface is None and block not in current_on_table):
                continue  # block not mentioned in goal
                
            if (goal_surface is None and block not in current_on_table) or \
               (current_surface is None and block not in self.goal_on_table) or \
               (goal_surface is not None and current_surface != goal_surface):
                h += 1  # basic mismatch cost
                
                # Additional cost if block is not clear (needs unstacking)
                if block not in current_clear:
                    # Estimate number of blocks above it
                    above = [b for b, under in current_on.items() if under == block]
                    h += len(above)
        
        # 2. Account for arm state
        if holding is not None:
            h += 1  # need to put down or stack the held block
        
        # 3. Additional cost for blocks that are in the way of goal stacks
        for block, under in self.goal_on.items():
            if under in current_on and current_on[under] != block:
                # The block that should be under is currently under something else
                h += 1
        
        return h
