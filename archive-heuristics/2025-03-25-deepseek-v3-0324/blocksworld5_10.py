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
    2. Adding penalties for blocks that are in towers but need to be moved
    3. Considering the arm state (whether we need to pick up or put down blocks)

    # Assumptions:
    - The arm can hold only one block at a time.
    - Blocks can only be stacked one on top of another.
    - The table has unlimited space for placing blocks.

    # Heuristic Initialization
    - Extract the goal conditions to know the desired final positions of blocks.
    - Build a mapping of which blocks should be on which other blocks or on the table.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position (on correct block or table)
    2. For blocks not in correct position:
       - If it's part of a tower that needs to be moved, add extra cost
       - If it's being held, check if it needs to be placed correctly
    3. Count the number of blocks that need to be moved (each move requires at least 2 actions)
    4. Add 1 if the arm is empty but needs to pick up a block
    5. Add 1 if the arm is holding a block that's not in its final position
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: {block: (on-table/on, target)}
        self.goal_structure = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                self.goal_structure[parts[1]] = ("on", parts[2])
            elif parts[0] == "on-table":
                self.goal_structure[parts[1]] = ("on-table", None)
            elif parts[0] == "clear":
                # Clear goals are handled implicitly by on/on-table
                continue

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        cost = 0
        
        # Track current block positions and arm state
        current_positions = {}
        holding_block = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_positions[parts[1]] = ("on", parts[2])
            elif parts[0] == "on-table":
                current_positions[parts[1]] = ("on-table", None)
            elif parts[0] == "holding":
                holding_block = parts[1]
            elif parts[0] == "arm-empty":
                arm_empty = True
        
        # Check each block's position against goal
        misplaced_blocks = set()
        for block, (current_type, current_target) in current_positions.items():
            if block not in self.goal_structure:
                continue  # Block not mentioned in goals
            
            goal_type, goal_target = self.goal_structure[block]
            
            if current_type != goal_type or current_target != goal_target:
                misplaced_blocks.add(block)
        
        # Basic cost: at least 2 actions per misplaced block (pickup + putdown/stack)
        cost += 2 * len(misplaced_blocks)
        
        # Additional cost for blocks that are in towers but need to be moved
        for block in misplaced_blocks:
            # If this block is supporting other blocks that also need to be moved
            # we'll need to unstack them first
            supporting_blocks = [b for b, (t, target) in current_positions.items() 
                               if t == "on" and target == block and b in misplaced_blocks]
            cost += len(supporting_blocks)
        
        # Handle arm state
        if holding_block:
            if holding_block in misplaced_blocks:
                # Need to place this block correctly
                cost += 1
            else:
                # Need to put it down to work on other blocks
                cost += 1
        elif arm_empty and misplaced_blocks:
            # Need to pick up a block
            cost += 1
        
        # If we're holding a block that's already in correct position,
        # we might need to put it down to work on other blocks
        if holding_block and holding_block not in misplaced_blocks and misplaced_blocks:
            cost += 1
        
        return cost
