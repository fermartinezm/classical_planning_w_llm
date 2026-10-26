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
    2. Considering whether the arm is empty or holding a block
    3. Accounting for necessary unstacking and stacking operations

    # Assumptions:
    - The goal state specifies exact block positions (on relations)
    - Blocks can only be moved one at a time
    - The arm can hold only one block at a time
    - Blocks must be clear to be moved

    # Heuristic Initialization
    - Extract goal conditions to determine correct block positions
    - No static facts are needed in Blocksworld

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position in the goal state
    2. For blocks not in correct position:
       - If the block is under another block, add 1 for unstacking
       - If the block needs to be placed under another block, add 1 for stacking
       - If the block is on the table but should be on another block, add 1 for pickup and stack
       - If the block is on another block but should be on table, add 1 for unstack and putdown
    3. Add 1 if the arm is currently holding a block that needs to be placed
    4. Add 1 if the arm is empty but needs to pick up a block
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal conditions into a more accessible format
        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear = set()
        
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "on":
                block, under = args
                self.goal_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                self.goal_on_table.add(block)
            elif predicate == "clear":
                block = args[0]
                self.goal_clear.add(block)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        cost = 0
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Track current block positions
        current_on = {}
        current_on_table = set()
        current_clear = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "on":
                block, under = args
                current_on[block] = under
            elif predicate == "on-table":
                block = args[0]
                current_on_table.add(block)
            elif predicate == "clear":
                block = args[0]
                current_clear.add(block)
            elif predicate == "holding":
                block = args[0]
                holding = block
            elif predicate == "arm-empty":
                arm_empty = True
        
        # For each block, check if it's in correct position
        for block in self.goal_on:
            # Block should be on another block in goal
            if block in current_on:
                if current_on[block] != self.goal_on[block]:
                    # Wrong position - need to move
                    cost += 1
                    # If something is on top, need to unstack first
                    if any(b for b in current_on if current_on[b] == block):
                        cost += 1
            elif block in current_on_table:
                # On table but should be on another block
                cost += 1  # pickup and stack
                # If something is on top in current state, need to unstack first
                if any(b for b in current_on if current_on[b] == block):
                    cost += 1
        
        for block in self.goal_on_table:
            # Block should be on table in goal
            if block in current_on:
                # On another block but should be on table
                cost += 1  # unstack and putdown
                # If something is on top in current state, need to unstack first
                if any(b for b in current_on if current_on[b] == block):
                    cost += 1
        
        # Account for arm state
        if holding:
            # If holding a block that's not in correct position, need to place it
            if (holding in self.goal_on and 
                (holding not in current_on or current_on[holding] != self.goal_on[holding])):
                cost += 1
            elif (holding in self.goal_on_table and 
                  holding not in current_on_table):
                cost += 1
        elif not arm_empty:
            # Arm is not empty but not holding anything (shouldn't happen)
            cost += 1
        else:
            # Arm is empty - might need to pick up a block
            # Find a block that needs to be moved and is clear
            for block in set(self.goal_on.keys()).union(self.goal_on_table):
                if (block in current_on and current_on[block] != self.goal_on.get(block, None)) or \
                   (block in current_on_table and block not in self.goal_on_table):
                    if block in current_clear:
                        cost += 1
                        break
        
        return cost
