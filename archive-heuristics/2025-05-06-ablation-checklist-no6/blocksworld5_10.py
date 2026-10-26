from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected format, e.g., return empty list or raise error
        # Assuming valid PDDL fact string format for this problem
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to correct block positions and free the arm.
    It counts blocks that are not in their correct goal position, blocks that are wrongly placed on others, and a busy arm.

    # Assumptions:
    - The goal state defines the desired stack configurations using `on` and `on-table` predicates.
    - Blocks not explicitly mentioned in the goal `on` or `on-table` facts are not considered for goal positioning (though they might act as obstacles).
    - A valid state representation explicitly lists all true facts regarding block positions (`on`, `on-table`, `holding`) and arm state (`arm-empty`, `holding`).

    # Heuristic Initialization
    - Parse the goal facts to build mappings representing the desired stack structure:
        - `goal_pos`: maps a block to the block it should be directly on top of, or 'table'.
        - `goal_above`: maps a block to the block that should be directly on top of it.
    - Identify all blocks explicitly mentioned in the goal `on` or `on-table` facts (`goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Parse the current state facts to determine the current position (on block, on table, or holding) for each block and the state of the arm (empty or holding). Store this in `current_pos` and `arm_empty`.
    2. Initialize the heuristic value `h` to 0.
    3. **Component 1 (Misplaced Blocks):** Iterate through each block `X` that has a specified goal position (i.e., is in `self.goal_pos`). If the current position of `X` (`current_pos.get(X)`) is different from its goal position (`self.goal_pos[X]`), increment `h` by 1.
    4. **Component 2 (Wrongly Placed Blocks on Top):** Iterate through all `(on Z Y)` facts in the current state. If `Y` is a block relevant to the goal structure (i.e., `Y` is in `self.goal_blocks`), check if `Z` is the block that should be on `Y` in the goal state (`self.goal_above.get(Y)`). If nothing should be on `Y` in the goal (`self.goal_above.get(Y)` is None) or if `Z` is not the block that should be on `Y` (`Z != self.goal_above.get(Y)`), increment `h` by 1.
    5. **Component 3 (Busy Arm):** If the arm is not empty (`arm_empty` is False), increment `h` by 1.
    6. Return the total value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        self.goals = task.goals

        # Store goal positions: block -> base_block or 'table'
        self.goal_pos = {}
        # Store inverse goal positions: base_block -> block_on_top
        self.goal_above = {}
        # Set of all blocks involved in the goal on/on-table facts
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_pos[block] = base
                    self.goal_above[base] = block
                    self.goal_blocks.add(block)
                    self.goal_blocks.add(base)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_pos[block] = 'table'
                    self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for building the structure.

        # Static facts are not used in this domain.
        # static_facts = task.static # frozenset() for blocksworld

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Extract current state information
        current_pos = {} # block -> base_block or 'table' or 'holding'
        arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    current_pos[block] = base
            elif predicate == 'on-table':
                if len(parts) == 2:
                    block = parts[1]
                    current_pos[block] = 'table'
            elif predicate == 'holding':
                 if len(parts) == 2:
                    block = parts[1]
                    current_pos[block] = 'holding' # Store what block is held
            elif predicate == 'arm-empty':
                arm_empty = True
            # Ignore 'clear' facts for heuristic calculation

        h = 0

        # Component 1: Blocks not on their correct base (or table)
        # Only consider blocks that have a specified goal position (on or on-table)
        for block, goal_base in self.goal_pos.items():
            # If a block is in goal_pos, it must be in current_pos in any valid state
            # (either on something, on table, or holding).
            # We use .get() to safely retrieve the current position, defaulting to None
            # if the block is not explicitly listed in the state facts (which shouldn't happen
            # in a valid state representation but is safer).
            current_base = current_pos.get(block)

            if current_base != goal_base:
                 h += 1

        # Component 2: Blocks that are wrongly placed on top of other blocks.
        # Iterate through current 'on' facts.
        for fact in state:
            parts = get_parts(fact)
            if not parts or parts[0] != 'on' or len(parts) != 3:
                continue # Not an 'on' fact or malformed

            block_on_top, block_below = parts[1], parts[2]

            # Check if the block below is relevant to the goal structure.
            # A block is relevant if it's part of any goal relationship (either as base or block_on_top).
            # This is covered by checking if block_below is in self.goal_blocks.
            if block_below in self.goal_blocks:
                 # Check what *should* be on block_below in the goal
                 goal_block_above = self.goal_above.get(block_below)

                 # If something is on block_below in the current state (block_on_top exists)
                 # AND either nothing should be on block_below in the goal (goal_block_above is None)
                 # OR the block on top is the wrong one (block_on_top != goal_block_above)
                 if goal_block_above is None or block_on_top != goal_block_above:
                     h += 1

        # Component 3: Arm is busy
        if not arm_empty:
            h += 1

        return h
