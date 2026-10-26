from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting several types of mismatches between the current state and the goal state:
    1. Blocks that are not on their correct base (or table) as specified in the goal, or are held when they should be on a base/table.
    2. Mismatches in the blocks stacked above other blocks: a block has the wrong block on top of it, or a block that should be clear has something on top of it, or a block that should have something specific on it has nothing on top.
    3. The robot arm is not empty when it should be in the goal.

    # Assumptions
    - The goal state defines a specific configuration of blocks stacked on the table.
    - The arm must be empty in the goal state (this is typical).
    - All blocks mentioned in the goal or initial state are relevant.
    - Every block in a valid state is either on another block, on the table, or held by the arm.

    # Heuristic Initialization
    - Parse the goal facts to determine the desired base for each block (`goal_base`), the block that should be on top of each block (`goal_above`), and if the arm should be empty (`goal_arm_empty`).
    - Collect all unique block objects involved in the problem from initial state and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts to determine the current base for each block (`current_base`), the block currently on top of each block (`current_above`), which block is held (`current_holding`), and if the arm is empty (`current_arm_empty`).
    2. Initialize heuristic value `h = 0`.
    3. Count blocks whose current base is wrong: Iterate through all blocks. If a block has a defined goal base (i.e., it's part of a goal stack or should be on the table), check if its current base (on another block, on the table, or held) matches the goal base. If they differ, increment `h`.
    4. Count mismatches for blocks above: Iterate through all potential bases (all blocks and the 'table'). Determine what block should be on this base in the goal (`goal_top`) and what block is currently on it (`current_top`). If `goal_top` is defined and `current_top` is different, increment `h`. If `goal_top` is None (meaning the base should be clear) and `current_top` is not None, increment `h`.
    5. Check the arm state: If the arm should be empty in the goal but is currently holding a block, increment `h`.
    6. Explicitly check if the state is the goal state. If it is, return 0. Otherwise, return the calculated `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and objects."""
        self.goals = task.goals

        # Parse goal facts
        self.goal_base = {} # block -> block_below or 'table'
        self.goal_above = {} # block_below -> block_above
        self.goal_arm_empty = False

        self.all_objects = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                self.goal_base[block] = base
                self.goal_above[base] = block
                self.all_objects.add(block)
                self.all_objects.add(base)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_base[block] = 'table'
                self.all_objects.add(block)
            elif predicate == 'clear':
                block = parts[1]
                # If (clear X) is a goal, it implies nothing should be on X.
                # This is captured by goal_above.get(X) being None.
                # Add X to objects if not already present.
                self.all_objects.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True

        # Collect objects from initial state as well, just in case some are only in init.
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 self.all_objects.update(parts[1:])
             elif parts[0] == 'arm-empty':
                 pass # arm-empty doesn't add objects

        # Remove 'table' from objects set if it was added
        self.all_objects.discard('table')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached. If so, heuristic must be 0.
        # This is a requirement: "The heuristic is 0 only for goal states."
        if self.goal_reached(state):
             return 0

        # Parse current state facts
        current_base = {} # block -> block_below or 'table'
        current_above = {} # block_below -> block_above
        current_holding = None
        current_arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, base = parts[1], parts[2]
                current_base[block] = base
                current_above[base] = block
            elif predicate == 'on-table':
                block = parts[1]
                current_base[block] = 'table'
            elif predicate == 'holding':
                current_holding = parts[1]
            elif predicate == 'arm-empty':
                current_arm_empty = True

        h = 0

        # Count blocks whose current base is wrong compared to their goal base or are held when they shouldn't be
        for block in self.all_objects:
            goal_b = self.goal_base.get(block)
            # Only consider blocks that have a specific goal base or are on the table in the goal
            if goal_b is not None:
                current_b = current_base.get(block)
                if current_holding == block:
                     current_b = 'arm' # Represent being held as a base 'arm'
                # If block is not in current_base and not held, it implies it's not on anything or table.
                # This shouldn't happen in valid blocksworld states derived from initial state.
                # If it somehow happens, current_b remains None. Let's assume valid states.

                if current_holding == block:
                     # If held, it's not on its goal base/table unless the goal is to be held (unlikely).
                     # Assuming 'arm' is never a goal base.
                     h += 1
                elif current_b != goal_b:
                     h += 1 # Block is on the wrong base/table.

        # Count mismatches for blocks above
        # Iterate through all objects that can potentially be a base (all objects + 'table')
        potential_bases = set(self.all_objects)
        potential_bases.add('table')

        for base in potential_bases:
            # What should be on this base in the goal?
            goal_top = self.goal_above.get(base) # Specific block, or None if base should be clear

            # What is currently on this base?
            current_top = current_above.get(base) # Specific block, or None if base is clear

            if goal_top is not None: # Base should have a specific block on it in the goal
                if current_top != goal_top:
                    h += 1 # Base has the wrong block on it (or nothing)
            else: # Base should be clear in the goal (goal_top is None)
                if current_top is not None:
                    h += 1 # Base is blocked when it should be clear.

        # Add cost for arm not being empty if it should be
        if self.goal_arm_empty and not current_arm_empty:
            h += 1 # Arm is holding something and should be empty.


        # If not a goal state, return the calculated heuristic value.
        # Ensure it's finite and non-negative.
        return h

    def goal_reached(self, state):
        """
        Check if the goal is reached. This is the same logic as Task.goal_reached,
        but included here to ensure the heuristic returns 0 iff the goal is reached.
        """
        # The goal has been reached if all facts that are true in "goals"
        # are true in "state".
        return set(self.goals) <= set(state)

