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
    2. Adding penalties for blocks that need to be moved to free other blocks
    3. Considering the arm state (whether we're holding a block that needs to be placed)

    # Assumptions:
    - The goal specifies a complete tower configuration (all blocks must be in specific positions)
    - Blocks can only be moved one at a time
    - The arm can hold only one block at a time

    # Heuristic Initialization
    - Extract the goal configuration (which blocks are on which other blocks or on table)
    - Build a mapping of goal positions for quick lookup

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on table in goal but not in current state: needs to be put down
       - If on another block in goal but not in current state: needs to be stacked
    2. For blocks that are in the way (supporting other blocks but not in goal position):
       - Add penalty for each block that needs to be moved to free them
    3. If the arm is holding a block:
       - If it's in the correct position already: no extra cost
       - Otherwise: add cost for placing it somewhere temporarily
    4. The total heuristic is the sum of:
       - Number of blocks not in correct position
       - Number of blocks that need to be moved to free correct positions
       - Additional cost if holding a block that's not in goal position
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal configuration
        self.goal_on = {}  # block -> what it's on in goal
        self.goal_under = {}  # block -> what's on it in goal
        self.goal_on_table = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_on[parts[1]] = parts[2]
                self.goal_under[parts[2]] = parts[1]
            elif parts[0] == "on-table":
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Track current configuration
        current_on = {}
        current_under = {}
        current_on_table = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
                current_under[parts[2]] = parts[1]
            elif parts[0] == "on-table":
                current_on_table.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
            elif parts[0] == "arm-empty":
                arm_empty = True
        
        h = 0
        
        # Count blocks not in correct position
        for block in self.goal_on:
            if block in current_on:
                if current_on[block] != self.goal_on[block]:
                    h += 1
            elif block in current_on_table and block not in self.goal_on_table:
                h += 1
        
        # Add blocks that are on table in goal but not currently
        for block in self.goal_on_table:
            if block not in current_on_table and (block not in current_on or current_on[block] != "table"):
                h += 1
        
        # Add penalty for blocks that are in the way
        for block in current_under:
            if block in self.goal_on and current_under[block] != self.goal_on[block]:
                h += 1
        
        # If holding a block that's not in its correct position
        if holding is not None:
            if (holding in self.goal_on and 
                (holding not in current_on or current_on[holding] != self.goal_on[holding])):
                h += 1
        
        return h
