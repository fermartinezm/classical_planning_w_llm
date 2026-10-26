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
    1. Counting blocks that are not in their correct position in the goal state
    2. Considering whether the arm is currently holding a block
    3. Accounting for necessary unstacking and stacking operations

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space for placing blocks

    # Heuristic Initialization
    - Extract the goal conditions to determine the desired block configuration
    - Build a mapping of which blocks should be on which other blocks in the goal state

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position according to the goal:
       - If not, add to the heuristic value
    2. If the arm is holding a block that's not in its correct position, add 1
    3. For each block that needs to be moved:
       - If it's under another block, add 1 for the unstack operation
       - If it needs to be placed under another block, add 1 for the stack operation
    4. For blocks that need to be moved to/from the table, add 1 for putdown/pickup
    5. The total heuristic is the sum of all these necessary operations
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
        h = 0
        
        # Check if arm is holding a block
        holding_block = None
        for fact in state:
            if match(fact, "holding", "*"):
                holding_block = get_parts(fact)[1]
                break
        
        # Build current block structure
        current_on = {}  # block: what it's on
        current_clear = set()  # blocks that are clear
        current_on_table = set()  # blocks on table
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                current_on[parts[1]] = parts[2]
            elif parts[0] == "on-table":
                current_on_table.add(parts[1])
            elif parts[0] == "clear":
                current_clear.add(parts[1])
        
        # For each block, check if it's in correct position
        for block, (goal_type, goal_target) in self.goal_structure.items():
            if goal_type == "on":
                # Block should be on another block in goal
                if block in current_on:
                    if current_on[block] != goal_target:
                        h += 1  # Needs to be moved
                else:
                    h += 1  # Not on anything (must be held or on table incorrectly)
            elif goal_type == "on-table":
                # Block should be on table in goal
                if block not in current_on_table:
                    h += 1  # Needs to be put on table
        
        # Additional cost if holding a block that's not in its correct position
        if holding_block:
            if holding_block in self.goal_structure:
                goal_type, goal_target = self.goal_structure[holding_block]
                if goal_type == "on":
                    # Check if we're holding over the correct block
                    # Since we can't see what's below, assume it's wrong
                    h += 1
                else:
                    # Should be on table but we're holding it
                    h += 1
        
        # Add cost for necessary unstack operations
        for block in current_on:
            if block in self.goal_structure:
                goal_type, goal_target = self.goal_structure[block]
                if goal_type == "on" and current_on[block] != goal_target:
                    # Block is on wrong thing, needs to be unstacked
                    h += 1
        
        # Ensure heuristic is 0 for goal states
        if h == 0 and self.goals <= state:
            return 0
        
        # Add base cost for each block that needs to be moved
        return max(1, h)
