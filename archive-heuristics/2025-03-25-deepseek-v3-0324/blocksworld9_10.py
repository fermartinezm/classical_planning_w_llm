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
    - The goal state specifies exact positions for all relevant blocks.

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired positions of blocks.
    - Build a mapping of which blocks should be on which other blocks in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on the correct block (or table) and all blocks above it are correct, no action needed.
       - If not, we'll need at least one action to correct it.
    2. For blocks that need to be moved:
       - If the block is buried under others, we need to unstack all blocks above it first.
       - If the block is free but not in goal position, we need to move it.
    3. Account for the arm state:
       - If holding a block, we need to put it down before doing other operations.
    4. The total heuristic is the sum of:
       - The number of blocks not in their goal position
       - The number of blocks above them that need to be moved first
       - Plus 1 if the arm is holding a block (to account for the putdown action)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: maps each block to what it should be on
        self.goal_on = {}
        for goal in self.goals:
            if match(goal, "on", "*", "*"):
                _, block, under = get_parts(goal)
                self.goal_on[block] = under
            elif match(goal, "on-table", "*"):
                _, block = get_parts(goal)
                self.goal_on[block] = "table"

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Check if arm is holding a block
        holding_block = any(match(fact, "holding", "*") for fact in state)
        
        # Build current block structure
        current_on = {}
        on_table = set()
        clear_blocks = set()
        
        for fact in state:
            if match(fact, "on", "*", "*"):
                _, block, under = get_parts(fact)
                current_on[block] = under
            elif match(fact, "on-table", "*"):
                _, block = get_parts(fact)
                on_table.add(block)
            elif match(fact, "clear", "*"):
                _, block = get_parts(fact)
                clear_blocks.add(block)
        
        # Calculate heuristic value
        h = 0
        
        # If holding a block, we need at least one action to put it down
        if holding_block:
            h += 1
            
        # For each block that has a goal position
        for block, goal_under in self.goal_on.items():
            # Check if block is in correct position
            current_under = current_on.get(block, "table" if block in on_table else None)
            
            if current_under != goal_under:
                # Block needs to be moved - count this block
                h += 1
                
                # If it's not clear, we need to unstack blocks above it first
                if block not in clear_blocks:
                    # Count the number of blocks above this one
                    above = 0
                    current = block
                    while current in current_on:
                        current = current_on[current]
                        above += 1
                    h += above
        
        return h
