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
    2. Adding the number of blocks that need to be moved to free required blocks
    3. Considering the current arm state (holding a block or empty)

    # Assumptions:
    - The goal consists of a single tower of blocks (though may have multiple clear blocks)
    - Blocks can only be stacked one at a time
    - The arm can hold only one block at a time

    # Heuristic Initialization
    - Extract the goal tower structure from the goal conditions
    - Identify the base block of the goal tower (on-table)
    - Build a mapping of each block to its required supporting block in the goal

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal tower structure:
       - Find the base block (on-table in goal)
       - Build the chain of blocks above it (on relations)
    2. For each block in the current state:
       - If it's part of the goal tower but not in the correct position: +1 move
       - If it's blocking a block that needs to be moved: +1 move
    3. If the arm is holding a block that's not part of the goal tower: +1 putdown
    4. If the arm is empty but needs to pick up a block: +1 pickup
    5. For each block that needs to be moved to free another block: +1 move per block
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal tower structure: maps each block to what it should be on
        self.goal_on = {}
        self.goal_clear = set()
        self.goal_on_table = set()
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "clear":
                self.goal_clear.add(args[0])
            elif predicate == "on-table":
                self.goal_on_table.add(args[0])
        
        # Find the base of the goal tower (on-table in goal)
        self.goal_tower_base = None
        for block in self.goal_on_table:
            if block in self.goal_on.values():  # Has something on it in goal
                self.goal_tower_base = block
                break

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Build current on relations
        current_on = {}
        current_clear = set()
        current_on_table = set()
        holding = None
        
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under = args
                current_on[block] = under
            elif predicate == "clear":
                current_clear.add(args[0])
            elif predicate == "on-table":
                current_on_table.add(args[0])
            elif predicate == "holding":
                holding = args[0]
        
        # Count mismatches in the goal tower
        misplaced_blocks = 0
        blocks_to_move = set()
        
        # Check the goal tower from base up
        if self.goal_tower_base:
            current_block = self.goal_tower_base
            while current_block in self.goal_on:
                next_block = [k for k, v in self.goal_on.items() if v == current_block][0]
                
                # Check if next_block is on current_block in current state
                if next_block in current_on and current_on[next_block] == current_block:
                    pass  # Correct position
                else:
                    misplaced_blocks += 1
                    blocks_to_move.add(next_block)
                
                current_block = next_block
        
        # Count blocks that are blocking required blocks
        blocking_blocks = 0
        for block in blocks_to_move:
            if block in current_on:  # Something is on this block
                blocking_blocks += 1
        
        # Check if arm is holding a block that's not part of the goal
        arm_penalty = 0
        if holding is not None:
            if holding not in self.goal_on and holding not in self.goal_on_table:
                arm_penalty = 1  # Need to put this down
        
        # Estimate total actions needed
        return misplaced_blocks + blocking_blocks + arm_penalty
