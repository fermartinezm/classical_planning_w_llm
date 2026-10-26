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
    3. Accounting for the need to clear blocks before moving them

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - The goal specifies a complete configuration (all blocks must be in some position)

    # Heuristic Initialization
    - Extract the goal conditions into a dictionary mapping each block to its required position
    - Build a goal dependency graph to understand block ordering constraints

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the current state
    2. For blocks not in correct position:
       - If the block is on the table but needs to be on another block, add 1 action (pickup + stack)
       - If the block is on another block but needs to be on the table, add 1 action (unstack + putdown)
       - If the block is on the wrong supporting block, add 2 actions (unstack + stack)
    3. For each block that needs to be moved, check if its current supporting block is correct:
       - If not, we may need additional moves to free the supporting block first
    4. Add 1 for each block that needs to be moved but is currently not clear
    5. If the arm is holding a block that's not in its goal position, add 1 action to put it down
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal configuration: maps each block to its required position
        self.goal_config = {}
        # Build goal dependency graph: maps each block to what should be below it in goal
        self.goal_below = {}
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                self.goal_config[block] = under
                self.goal_below[block] = under
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_config[block] = "table"
                self.goal_below[block] = None

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Build current configuration
        current_config = {}
        current_clear = set()
        holding = None
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                current_config[block] = under
            elif parts[0] == "on-table":
                block = parts[1]
                current_config[block] = "table"
            elif parts[0] == "clear":
                current_clear.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
        
        h = 0
        
        # Check each block's position
        for block in self.goal_config:
            goal_pos = self.goal_config[block]
            current_pos = current_config.get(block, None)
            
            # Block is in correct position
            if goal_pos == current_pos:
                continue
                
            # Block needs to be moved
            h += 1  # At least one action (pickup/putdown/stack/unstack)
            
            # Additional actions may be needed if:
            # 1. The block is not clear (need to unstack what's on top)
            if block not in current_clear:
                h += 1
                
            # 2. The target position is not clear (for stacking)
            if goal_pos != "table" and goal_pos not in current_clear:
                h += 1
                
            # 3. The block is on another block that needs to be moved first
            if current_pos != "table" and current_pos in self.goal_config:
                if current_config.get(current_pos, None) != self.goal_config.get(current_pos, None):
                    h += 1
        
        # If holding a block that's not in goal position, need to put it down
        if holding and holding in self.goal_config:
            if current_config.get(holding, None) != self.goal_config[holding]:
                h += 1
        
        return h
