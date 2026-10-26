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
    2. Adding the number of blocks that need to be moved to clear blocks below them
    3. Considering the current state of the arm (holding a block or empty)

    # Assumptions:
    - The goal consists of one or more towers of blocks (each tower ending with a block on table)
    - Each block can be part of only one tower in the goal state
    - The heuristic doesn't need to be admissible (can overestimate)

    # Heuristic Initialization
    - Extract the goal structure (towers of blocks) from the goal conditions
    - For each block, store its correct "on" relationship and whether it should be on table

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block in the current state:
       a. If it's being held, add 1 action (must put it down somewhere)
       b. If it's not in its correct position (either wrong "on" relation or wrong table status):
          - Add 2 actions (pick up and put down in correct position)
          - If the block below it is also wrong, add 1 more action (need to clear it first)
    2. For blocks that are in correct position but have incorrect blocks above them:
       - Add 1 action per incorrect block above (need to move them to clear the correct block)
    3. If the arm is holding a block that's already in correct position, add 1 action (putdown)
    4. If the arm is empty but should be holding a block (for optimal solution), add 1 action
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: maps each block to what it should be on (or None for table)
        self.goal_on = {}
        self.goal_clear = set()
        
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
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Track which blocks are in correct position
        correct_blocks = set()
        # Count of blocks that need to be moved
        move_count = 0
        # Blocks that are correctly placed but have incorrect blocks above them
        correct_but_covered = 0
        # Check if arm is holding a block
        holding_block = any(match(fact, "holding", "*") for fact in state)
        
        # First pass: check current block positions
        for fact in state:
            predicate, *args = get_parts(fact)
            
            if predicate == "on":
                block, under = args
                # Check if this is correct in goal
                if self.goal_on.get(block) == under:
                    correct_blocks.add(block)
                else:
                    move_count += 2  # pickup and putdown
                    # Check if the block below needs to be moved too
                    if under in self.goal_on and self.goal_on[under] != block:
                        move_count += 1  # need to clear the under block
                        
            elif predicate == "on-table":
                block = args[0]
                # Check if this block should be on table in goal
                if self.goal_on.get(block) is None:
                    correct_blocks.add(block)
                else:
                    move_count += 2  # pickup and stack
                    
            elif predicate == "clear":
                block = args[0]
                # Check if this block should be clear in goal
                if block in self.goal_clear:
                    correct_blocks.add(block)
                    
        # Second pass: check for correct blocks with incorrect blocks above them
        for fact in state:
            if match(fact, "on", "*", "*"):
                _, under, _ = get_parts(fact)
                if under in correct_blocks:
                    correct_but_covered += 1
        
        # If holding a block, we need at least one action to put it down
        if holding_block:
            move_count += 1
            
        # Total heuristic is sum of all identified necessary actions
        return move_count + correct_but_covered
