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
    This heuristic estimates the number of actions needed to transform the current
    block configuration into the goal configuration. It considers:
    - Blocks that are in the wrong position (not on their correct base)
    - Blocks that need to be moved to free other blocks
    - The current state of the arm (holding a block or empty)

    # Assumptions:
    - The goal specifies a complete configuration (all blocks have defined positions)
    - Blocks can only be moved one at a time
    - The gripper can hold only one block at a time

    # Heuristic Initialization
    - Extract the goal configuration (which blocks are on which other blocks or table)
    - Build a mapping of each block to its correct base (another block or table)
    - Identify which blocks should be clear in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position:
       - If on the correct base and the base is correct, no action needed
       - If on wrong base or table when it should be on a block, count 1 action
    2. For blocks that are blocking others:
       - If a block is above another block that needs to be moved, count 1 action
    3. Consider the current state of the arm:
       - If holding a block that's not in its correct position, count 1 action
    4. For each block that needs to be moved:
       - Count 1 action to pick it up (if not already held)
       - Count 1 action to put it down (stack or put on table)
    5. For blocks that need to be moved but are currently supporting others:
       - Count additional actions to unstack blocks above them
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal configuration: maps each block to what it should be on
        self.goal_on = {}
        # Blocks that should be clear in the goal state
        self.goal_clear = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                self.goal_on[parts[1]] = "table"
            elif parts[0] == "clear":
                self.goal_clear.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Current block positions
        current_on = {}
        current_clear = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                current_on[parts[1]] = "table"
            elif parts[0] == "clear":
                current_clear.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
            elif parts[0] == "arm-empty":
                arm_empty = True
        
        # Blocks that are in the wrong position
        wrong_positions = 0
        # Blocks that are blocking others that need to be moved
        blocking = 0
        
        # Check each block's position
        for block, correct_base in self.goal_on.items():
            current_base = current_on.get(block, None)
            
            # Block is in wrong position
            if current_base != correct_base:
                wrong_positions += 1
                
                # If it's on something that's not its correct base, that's blocking
                if current_base != "table" and current_base in self.goal_on:
                    blocking += 1
        
        # Check clear blocks
        for block in self.goal_clear:
            if block not in current_clear:
                blocking += 1
        
        # If holding a block that's not in its correct position, count as action
        holding_penalty = 0
        if holding is not None and holding in self.goal_on:
            if current_on.get(holding, None) != self.goal_on[holding]:
                holding_penalty = 1
        
        # Each block that needs to be moved requires at least 2 actions (pick and place)
        # Plus potential unstack actions for blocks above it
        estimate = 2 * wrong_positions + blocking + holding_penalty
        
        # If the arm is empty and we have blocks to move, we might need an extra action
        if arm_empty and wrong_positions > 0:
            estimate += 1
            
        return estimate
