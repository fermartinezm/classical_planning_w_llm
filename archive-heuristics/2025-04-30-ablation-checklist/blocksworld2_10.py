from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# No need for a general match function if we use get_parts and check predicate names directly
# def match(fact, *args): ...

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the number of actions needed to reach the goal
    by counting discrepancies between the current state and the goal state
    regarding block positions and stack structures. It is designed for
    greedy best-first search and does not need to be admissible.

    The heuristic value is the sum of the following components:
    1. For each block B that is part of an 'on' or 'on-table' goal:
       +1 if B is not currently in its goal position (i.e., not on the
          correct block or not on the table as required by the goal).
    2. For each block B that is part of an 'on' or 'on-table' goal AND
       is currently in its goal position:
       +1 if the block immediately on top of B in the current state is
          not the block that should be immediately on top of B according
          to the goal. This includes cases where something is on top but
          shouldn't be, or nothing is on top but something should be.
    3. For each block B that must be clear according to a goal:
       +1 if B is not clear in the current state.
    4. +1 if the arm must be empty according to a goal and is not empty
       in the current state.

    This heuristic captures the cost of fixing misplaced blocks, clearing
    incorrectly stacked blocks, and satisfying explicit clear/arm-empty goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Parse goal facts to build target configuration maps and sets
        self.goal_pos = {} # block -> block_below or 'table'
        self.goal_above = {} # block_below -> block_on_top
        self.goal_clear_blocks = set() # blocks that must be clear
        self.goal_arm_empty = False
        self.goal_blocks = set() # blocks mentioned in on/on-table goals

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block_on_top, block_below = parts[1], parts[2]
                self.goal_pos[block_on_top] = block_below
                self.goal_above[block_below] = block_on_top
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_pos[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear_blocks.add(block)
            elif predicate == "arm-empty":
                self.goal_arm_empty = True
            # Ignore other goal predicates if any (e.g., holding, which is not a standard BW goal)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Parse current state facts to build current configuration maps and sets
        current_pos = {} # block -> block_below or 'table' or 'holding'
        current_above = {} # block_below -> block_on_top
        state_clear_blocks = set()
        state_arm_empty = False
        # holding_block = None # Not strictly needed for this heuristic logic

        # Build maps and sets from state facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block_on_top, block_below = parts[1], parts[2]
                current_pos[block_on_top] = block_below
                current_above[block_below] = block_on_top
            elif predicate == "on-table":
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == "holding":
                block = parts[1]
                current_pos[block] = 'holding'
                # holding_block = block # Not used in this logic
            elif predicate == "clear":
                block = parts[1]
                state_clear_blocks.add(block)
            elif predicate == "arm-empty":
                state_arm_empty = True

        h = 0

        # Component 1 & 2: Misplaced blocks or blocks with wrong block immediately above
        # We only care about blocks explicitly mentioned in 'on' or 'on-table' goals
        for block in self.goal_blocks:
            b_goal_pos = self.goal_pos.get(block) # This should always exist for blocks in goal_blocks
            b_current_pos = current_pos.get(block) # Can be None if block isn't on/on-table/holding (invalid state?)

            # In a valid Blocksworld state, every block is either on another block,
            # on the table, or held. So, if a block from goal_blocks exists,
            # it should have an entry in current_pos unless the state is malformed.
            # Assuming valid states:
            if b_current_pos != b_goal_pos:
                h += 1 # Component 1: Block is in the wrong position
            else: # b_current_pos == b_goal_pos
                # Component 2: Block is in the correct position, but check the block above it
                block_above_b_goal = self.goal_above.get(block)
                block_above_b_current = current_above.get(block)
                if block_above_b_current != block_above_b_goal:
                    h += 1 # Wrong block immediately above (includes cases where one is None and the other isn't)

        # Component 3: Blocks that need to be clear but aren't
        for block in self.goal_clear_blocks:
            if block not in state_clear_blocks:
                 h += 1

        # Component 4: Arm needs to be empty but isn't
        if self.goal_arm_empty and not state_arm_empty:
             h += 1

        return h
