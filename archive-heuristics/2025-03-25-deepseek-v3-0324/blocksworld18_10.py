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
    1. Counting mismatched blocks (blocks not in their correct position)
    2. Adding the number of blocks that need to be moved to free other blocks
    3. Considering the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The goal specifies a complete tower configuration (all blocks must be in specific positions)
    - Blocks can only be moved one at a time
    - The arm can hold only one block at a time

    # Heuristic Initialization
    - Extract the goal configuration into a dictionary mapping blocks to their target positions
    - Identify which blocks should be on the table in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: +1 (need to put down)
       - If stacked in goal but not in current state: +1 (need to stack)
    2. For blocks that are blocking others from their correct positions:
       - Each such block adds +1 to the heuristic (need to move it away)
    3. Consider the arm state:
       - If holding a block that's not in its correct position: +1 (need to place it)
       - If arm is empty but needs to pick up a block: +1
    4. The total heuristic is the sum of these mismatches and required moves
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal configuration
        self.goal_on = {}  # Maps each block to what it should be on
        self.goal_on_table = set()  # Blocks that should be on table
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                self.goal_on_table.add(block)
            elif predicate == "clear":
                # Clear goals are handled implicitly by on/on-table
                pass

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Track current configuration
        current_on = {}  # What each block is currently on
        current_on_table = set()  # Blocks currently on table
        holding = None  # Currently held block
        arm_empty = False
        
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
            elif predicate == "arm-empty":
                arm_empty = True
        
        heuristic = 0
        
        # Check for each block if it's in correct position
        for block in self.goal_on:
            # Block should be on another block in goal
            if block in current_on:
                if current_on[block] != self.goal_on[block]:
                    heuristic += 1  # Needs to be moved
            else:
                heuristic += 1  # Not in correct position (should be stacked but isn't)
        
        for block in self.goal_on_table:
            if block not in current_on_table:
                heuristic += 1  # Should be on table but isn't
        
        # Check for blocks that are blocking others
        for block in current_on:
            under_block = current_on[block]
            if under_block in self.goal_on and self.goal_on[under_block] != block:
                # The under_block needs to be under something else
                heuristic += 1  # Need to move the blocking block
        
        # Handle arm state
        if holding:
            # If holding a block that's not in correct position
            if (holding in self.goal_on and (holding not in current_on or 
                                           current_on[holding] != self.goal_on[holding])) or \
               (holding in self.goal_on_table and holding not in current_on_table):
                heuristic += 1  # Need to place it
        elif not arm_empty:
            # Arm is not empty but not holding anything (shouldn't happen)
            heuristic += 1
        else:
            # Arm is empty - might need to pick up a block
            # Count blocks that need to be moved but are not clear
            for block in current_on:
                if (block in self.goal_on and current_on[block] != self.goal_on[block]) or \
                   (block in self.goal_on_table and block not in current_on_table):
                    if not any(fact.startswith(f"(clear {block})") for fact in state):
                        heuristic += 1  # Need to unstack something above it
        
        return heuristic
