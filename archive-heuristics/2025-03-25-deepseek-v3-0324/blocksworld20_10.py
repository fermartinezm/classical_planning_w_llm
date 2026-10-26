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
    2. Considering the dependencies between blocks (a block can't be placed correctly until its supporting block is correct)
    3. Accounting for the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - All blocks must be clear (no blocks on top) to be moved

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block configuration
    - Build a mapping of which blocks should be on which other blocks in the goal state
    - Identify which blocks should be on the table in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state, count as mismatch
       - If should be on another block but isn't, count as mismatch
    2. For blocks that are in correct position but have incorrect blocks below them:
       - Add penalty for each incorrect block in the stack below
    3. If the arm is holding a block:
       - If it's the correct block to place next, don't add penalty
       - Otherwise, add penalty for needing to put it down
    4. The total heuristic is the sum of:
       - Number of blocks not in correct position
       - Number of blocks that need to be moved to free required blocks
       - 1 if the arm is holding a block that's not needed next
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: what should be on what
        self.goal_on = {}  # block -> what it should be on
        self.goal_on_table = set()  # blocks that should be on table
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Track current block positions
        current_on = {}
        current_on_table = set()
        holding = None
        clear_blocks = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                current_on_table.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
            elif parts[0] == "clear":
                clear_blocks.add(parts[1])
        
        h = 0
        
        # Check blocks that should be on table
        for block in self.goal_on_table:
            if block not in current_on_table:
                h += 1
        
        # Check blocks that should be on other blocks
        for block, under_block in self.goal_on.items():
            if current_on.get(block) != under_block:
                h += 1
        
        # Add penalty for blocks that are in correct position but have wrong blocks below them
        for block in self.goal_on:
            if block in current_on and current_on[block] == self.goal_on[block]:
                # Check the stack below this block
                current_under = current_on.get(self.goal_on[block])
                goal_under = self.goal_on.get(self.goal_on[block])
                if current_under != goal_under:
                    h += 1
        
        # If holding a block that's not in its correct position, add penalty
        if holding is not None:
            if holding in self.goal_on_table and holding not in current_on_table:
                pass  # we're holding a block that needs to be on table
            elif holding in self.goal_on and current_on.get(holding) != self.goal_on[holding]:
                pass  # we're holding a block that needs to be on another block
            else:
                h += 1  # need to put this block down before we can work on others
        
        return h
