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
    1. Counting blocks that are not in their correct position in the goal tower(s)
    2. Considering whether the arm is empty or holding a block
    3. Accounting for necessary unstack/stack operations to correct block positions

    # Assumptions:
    - The goal consists of one or more towers of blocks
    - Each block can only be in one place in the goal state
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract the goal structure (which blocks are on which other blocks or on table)
    - Create a mapping from each block to its goal position

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: need to put it down
       - If on another block in goal but not in current state: need to stack it
    2. Count blocks that are not in their correct position (misplaced)
    3. For blocks that are in correct position but have incorrect blocks below them:
       - Need to unstack blocks above them to fix the tower
    4. If the arm is holding a block that's not in its correct position: +1 action
    5. Each misplaced block requires at least 2 actions (pick and place)
    6. Additional actions may be needed to clear blocks that are in the way
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: maps each block to what it should be on
        self.goal_on = {}
        self.goal_on_table = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        h = 0
        
        # Check if we're holding any block
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
                current_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                current_on_table.add(parts[1])
            elif parts[0] == "clear":
                clear_blocks.add(parts[1])
        
        # Count misplaced blocks
        misplaced = 0
        blocks_to_move = set()
        
        # Check blocks that should be on table
        for block in self.goal_on_table:
            if block not in current_on_table:
                if holding_block != block:  # If not already being moved
                    blocks_to_move.add(block)
        
        # Check blocks that should be on other blocks
        for block, under in self.goal_on.items():
            if current_on.get(block) != under:
                if holding_block != block:  # If not already being moved
                    blocks_to_move.add(block)
        
        # Count blocks that are in correct position but have wrong blocks below them
        for block, under in self.goal_on.items():
            if current_on.get(block) == under:
                # Check if the block below is correct
                if under in self.goal_on and current_on.get(under) != self.goal_on[under]:
                    blocks_to_move.add(under)
        
        # Each block to move requires at least 2 actions (pick and place)
        h += 2 * len(blocks_to_move)
        
        # If we're holding a block that needs to be placed, add 1 action
        if holding_block and holding_block in blocks_to_move:
            h += 1
        
        # Additional penalty for blocks that need to be cleared first
        for block in blocks_to_move:
            if block in current_on and not match(f"(clear {block})", *state):
                h += 1  # Need to unstack block above first
        
        return h
