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
    1. Counting mismatched blocks in their current positions vs goal positions
    2. Considering the need to clear blocks below mismatched blocks
    3. Accounting for the arm state (holding or empty)
    4. Adding penalties for blocks that need to be moved to/from the table

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be stacked one on top of another
    - The table has unlimited space
    - All blocks must be accounted for in both initial and goal states

    # Heuristic Initialization
    - Extract goal conditions into a dictionary mapping blocks to their desired positions
    - Identify which blocks should be on the table in the goal state
    - Build a goal structure representation for easy comparison

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if its current position matches the goal:
       - If on wrong block or wrong table status, count as mismatch
    2. For each mismatched block:
       - Add 1 for the move needed to correct its position
       - If it's currently on another block that needs to stay, add 1 for unstack
       - If it needs to go on another block that's not ready, add 1 for stack
    3. If the arm is holding a block that's not in its goal position, add 1 for putdown
    4. If a block is clear in goal but not currently, add 1 for clearing it
    5. If the goal requires a block to be on table but it's not, add 1 for putdown
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: {block: (on-table/on-block, target)}
        self.goal_structure = {}
        self.goal_on_table = set()
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under_block = args
                self.goal_structure[block] = ("on", under_block)
            elif predicate == "on-table":
                block = args[0]
                self.goal_structure[block] = ("on-table", None)
                self.goal_on_table.add(block)
            # Clear goals are handled during heuristic computation

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        cost = 0
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Check if arm is holding something not in goal position
        holding_block = None
        for fact in state:
            if match(fact, "holding", "*"):
                holding_block = get_parts(fact)[1]
                if holding_block in self.goal_structure:
                    goal_type, goal_target = self.goal_structure[holding_block]
                    if goal_type == "on":
                        # Need to stack it on goal_target
                        cost += 1
                    else:
                        # Need to put it on table
                        cost += 1
                break
        
        # Build current structure
        current_on = {}  # block: what it's on
        current_clear = set()  # blocks that are clear
        current_on_table = set()  # blocks on table
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, under_block = parts[1], parts[2]
                current_on[block] = under_block
            elif parts[0] == "on-table":
                block = parts[1]
                current_on_table.add(block)
            elif parts[0] == "clear":
                block = parts[1]
                current_clear.add(block)
        
        # Check each block's position
        for block in self.goal_structure:
            goal_type, goal_target = self.goal_structure[block]
            
            # Check current position
            if block in current_on:
                current_under = current_on[block]
                current_type = "on"
            else:
                current_under = None
                current_type = "on-table"
                current_on_table.add(block)  # in case it wasn't recorded yet
            
            # Position mismatch
            if (goal_type != current_type) or (goal_type == "on" and goal_target != current_under):
                cost += 1  # at least one action to fix
                
                # Additional cost if we need to unstack something first
                if current_type == "on" and current_under not in self.goal_on_table:
                    cost += 1
                
                # Additional cost if target block isn't ready
                if goal_type == "on" and goal_target not in current_clear:
                    cost += 1
        
        # Check clear conditions
        for goal in self.goals:
            if match(goal, "clear", "*"):
                block = get_parts(goal)[1]
                if block not in current_clear:
                    cost += 1  # need to clear this block
        
        return cost
