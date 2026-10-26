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
    - The goal specifies exact positions for blocks (on-table or on another block).

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired positions of each block.
    - Build a mapping of which blocks should be on which other blocks in the goal state.
    - Identify which blocks should be on the table in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on-table in goal but not in current state: needs to be moved to table
       - If on another block in goal but not in current state: needs to be stacked
    2. For blocks that need to be moved:
       - If the block is buried under others, we need to unstack all blocks above it first
       - If the target position is occupied, we need to clear it first
    3. Count the number of necessary moves:
       - Each block not in final position requires at least one move
       - Each block that needs to be cleared from above requires additional moves
       - Each block that needs to clear its target position requires additional moves
    4. Consider the arm state:
       - If holding a block that's not in its goal position, we need to put it down first
    5. The total heuristic is the sum of all these required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: what should be on what
        self.goal_on = {}  # block -> what it should be on (None means on-table)
        self.goal_clear = set()  # blocks that should be clear in goal
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                self.goal_on[block] = None
            elif predicate == "clear":
                block = args[0]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Build current block positions
        current_on = {}  # block -> what it's currently on (None means on-table)
        current_under = {}  # block -> what's on top of it
        current_clear = set()  # blocks that are currently clear
        
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under = args
                current_on[block] = under
                current_under[under] = block
            elif predicate == "on-table":
                block = args[0]
                current_on[block] = None
            elif predicate == "clear":
                block = args[0]
                current_clear.add(block)
            elif predicate == "holding":
                holding_block = args[0]
                # If holding a block that's not in goal position, need to put it down
                if (holding_block in self.goal_on and 
                    (holding_block not in current_on or 
                     current_on[holding_block] != self.goal_on[holding_block])):
                    return 1 + self._compute_heuristic(state, current_on, current_under, current_clear)
        
        return self._compute_heuristic(state, current_on, current_under, current_clear)
    
    def _compute_heuristic(self, state, current_on, current_under, current_clear):
        """Compute the heuristic value given the current state information."""
        h = 0
        
        # Check each block's position
        for block in self.goal_on:
            goal_under = self.goal_on[block]
            current_under_block = current_on.get(block, None)
            
            # Block is in wrong position
            if current_under_block != goal_under:
                h += 1  # at least one move needed
                
                # If block is buried under others, need to unstack them first
                if block in current_under:
                    h += 1  # unstack action for each block above
                    
                # If target position is occupied, need to clear it first
                if goal_under is not None and goal_under in current_under:
                    h += 1  # unstack action to clear target
                elif goal_under is None and goal_under in current_under:
                    h += 1  # unstack action to clear target
                    
        # Check clear conditions
        for block in self.goal_clear:
            if block not in current_clear:
                h += 1  # need to unstack block above
                
        return h
