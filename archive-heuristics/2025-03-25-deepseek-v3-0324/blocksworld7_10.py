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
    1. Counting mismatched blocks in the current state compared to the goal state
    2. Considering the dependencies between blocks (a block can't be placed correctly until its supporting block is correct)
    3. Accounting for the arm state (whether it's empty or holding a block)
    4. Estimating the number of unstack/stack operations needed to correct the tower structures

    # Assumptions:
    - The arm can hold only one block at a time
    - Blocks can only be moved when they are clear (no blocks on top)
    - The table has unlimited space
    - The goal state specifies exact positions for all blocks (complete specification)

    # Heuristic Initialization
    - Extract goal conditions and build a mapping of desired block positions
    - Identify which blocks should be on the table and which should be stacked
    - Build a dependency graph of blocks (which blocks need to be correct before others can be placed)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each block, check if it's in its correct position (either on table or on correct block)
    2. For incorrectly placed blocks:
       - If it's part of a stack that needs to be built, count the number of blocks above it that need to be moved
       - If it's on the wrong supporting block, count the moves needed to free it and place it correctly
    3. If the arm is holding a block, count the necessary putdown/stack action
    4. For each block that needs to be moved:
       - Add 1 for pickup/unstack
       - Add 1 for putdown/stack
       - If it needs to be moved to/from a stack, add moves for clearing blocks above it
    5. The total heuristic is the sum of all these estimated actions
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        self.static = task.static
        
        # Build goal structure: what block should be on what
        self.goal_on = {}  # block -> what it should be on
        self.goal_under = {}  # block -> what should be on it
        self.goal_on_table = set()
        
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                self.goal_on[block] = under
                if under not in self.goal_under:
                    self.goal_under[under] = []
                self.goal_under[under].append(block)
            elif parts[0] == "on-table":
                self.goal_on_table.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Current block positions
        current_on = {}
        current_under = {}
        current_on_table = set()
        holding = None
        arm_empty = False
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
                if under not in current_under:
                    current_under[under] = []
                current_under[under].append(block)
            elif parts[0] == "on-table":
                current_on_table.add(parts[1])
            elif parts[0] == "holding":
                holding = parts[1]
            elif parts[0] == "arm-empty":
                arm_empty = True
        
        # Calculate heuristic value
        h = 0
        
        # If holding a block, we need at least one action to place it
        if holding is not None:
            h += 1
        
        # Check each block's position
        for block in self.goal_on_table | set(self.goal_on.keys()):
            # Check if block is on table when it shouldn't be
            if block in current_on_table and block not in self.goal_on_table:
                h += 1  # pickup
                h += 1  # putdown/stack
            
            # Check if block is not on table when it should be
            elif block in self.goal_on_table and block not in current_on_table:
                h += 1  # unstack/putdown
            
            # Check if block is on wrong supporting block
            elif (block in self.goal_on and 
                  (block not in current_on or current_on[block] != self.goal_on[block])):
                h += 1  # unstack/pickup
                h += 1  # stack/putdown
            
            # Check if blocks above are incorrect (need to be moved to reach this one)
            if block in current_under:
                for above in current_under[block]:
                    if (above not in self.goal_under.get(block, []) or
                        (above in self.goal_on and self.goal_on[above] != block)):
                        h += 2  # pickup and putdown for each block above
        
        return h
