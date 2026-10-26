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
    2. Adding actions needed to clear blocks above incorrectly placed blocks
    3. Considering the arm's current state (holding a block or empty)

    # Assumptions:
    - The goal is always a single tower (though may have multiple clear blocks at top)
    - Blocks can only be stacked one at a time
    - The arm can hold only one block at a time

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block arrangement
    - Build a mapping of which block should be on which other block in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal structure by analyzing goal conditions:
       - Find which blocks should be on tables
       - Build a mapping of which blocks should be on top of others
    2. For each block in the current state:
       - If it's not where it should be in the goal structure, add to the cost:
         * 1 action to move it from its current position
         * 1 action to place it in the correct position
       - If blocks are above it that need to be moved first, add 2 actions per such block
    3. Special cases:
       - If the arm is holding a block, count the necessary putdown/stack action
       - If a block is in the correct position but has incorrect blocks above it, those need to be moved
    4. The total heuristic is the sum of all these required actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: maps each block to what it should be on in the goal
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
        cost = 0
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Check if arm is holding a block
        holding_block = None
        for fact in state:
            if match(fact, "holding", "*"):
                holding_block = get_parts(fact)[1]
                break
        
        # Build current block positions
        current_on = {}
        current_on_table = set()
        clear_blocks = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
            elif parts[0] == "on-table":
                block = parts[1]
                current_on_table.add(block)
            elif parts[0] == "clear":
                block = parts[1]
                clear_blocks.add(block)
        
        # For each block, check if it's in the correct position
        for block in self.goal_on:
            # Check if block is in correct position
            correct_under = self.goal_on[block]
            if block in current_on:
                if current_on[block] != correct_under:
                    cost += 2  # unstack + stack
            elif block in current_on_table and correct_under != "table":
                cost += 2  # pickup + stack
            
            # Check if blocks above need to be moved
            if block in current_on and current_on[block] != correct_under:
                # Need to move blocks above current position
                cost += 2  # per blocking block
        
        # Handle blocks that should be on table
        for block in self.goal_on_table:
            if block not in current_on_table:
                cost += 2  # unstack + putdown or pickup + putdown
        
        # If holding a block, need to place it
        if holding_block:
            cost += 1
        
        return cost
