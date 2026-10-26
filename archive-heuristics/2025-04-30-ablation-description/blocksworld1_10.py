from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The number of blocks that are part of the desired goal stack configuration
       but are not currently in their correct position relative to the block
       immediately below them (or the table).
    2. The number of goal predicates related to 'clear' status of blocks or
       the 'arm-empty' status that are not satisfied in the current state.

    # Assumptions:
    - The goal specifies the desired arrangement of blocks using 'on' and 'on-table' predicates,
      and potentially 'clear' and 'arm-empty'.
    - The heuristic is admissible as it counts necessary conditions that must be met,
      and each condition requires at least one action to resolve. The sum of admissible
      heuristics is admissible.
    - The heuristic is 0 if and only if the state is the goal state.

    # Heuristic Initialization
    - Parse the goal state to determine the desired position (block below or table)
      for each block that is part of the goal stack configuration ('on' or 'on-table' goals).
      Store this mapping in `self.goal_positions`.
    - Identify the set of blocks that are part of the goal stack configuration
      ('on' or 'on-table' goals). Store this set in `self.goal_blocks`.
    - Identify the set of blocks that need to be 'clear' in the goal state.
      Store this set in `self.goal_clear_blocks`.
    - Check if the 'arm-empty' predicate is a goal. Store this boolean in `self.goal_arm_empty`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize `misplaced_count` to 0. This will count blocks not in their correct 'on'/'on-table' position.
    2. Initialize `additional_cost` to 0. This will count unsatisfied 'clear' and 'arm-empty' goals.
    3. Parse the current state to determine the current position (block below, table, or hand)
       for every block. Store this mapping in `current_positions`. Also, identify the block
       currently being held, if any.
    4. Iterate through each block in `self.goal_blocks` (blocks whose 'on'/'on-table' position
       is specified in the goal).
    5. For each such block, retrieve its desired `goal_pos` from `self.goal_positions`
       and its `current_pos` from `current_positions`.
    6. If `current_pos` does not match `goal_pos`, increment `misplaced_count`. This handles
       cases where a block is on the wrong block, on the table when it should be on a block,
       on a block when it should be on the table, or currently held.
    7. Check if the 'arm-empty' predicate is a goal (`self.goal_arm_empty`) and if it is
       not present in the current state. If both are true, increment `additional_cost`.
    8. Identify the set of blocks that are currently 'clear' in the state.
    9. Iterate through each block in `self.goal_clear_blocks` (blocks that need to be clear
       in the goal). If a block is not in the set of currently clear blocks, increment
       `additional_cost`.
    10. The total heuristic value is the sum of `misplaced_count` and `additional_cost`.
        This sum is 0 if and only if the state is the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and conditions.
        """
        self.goals = task.goals

        # Component 1: Goal positions for blocks
        self.goal_positions = {}  # Maps block -> block_below or 'table'
        self.goal_blocks = set()  # Set of blocks mentioned in goal 'on'/'on-table' facts

        # Component 2: Other goal conditions
        self.goal_clear_blocks = set() # Set of blocks that need to be clear
        self.goal_arm_empty = False    # True if arm-empty is a goal

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]

            if predicate == 'on':
                block, below = parts[1], parts[2]
                self.goal_positions[block] = below
                self.goal_blocks.add(block)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_positions[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                block = parts[1]
                self.goal_clear_blocks.add(block)
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True

        # Ensure 'table' is not in goal_blocks if it was accidentally added
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if the goal is reached (heuristic is 0)
        if self.goals <= state:
             return 0

        # Component 1: Count blocks not in correct 'on'/'on-table' position
        current_positions = {} # Maps block -> block_below or 'table' or 'hand'
        currently_held = None

        # Identify current positions and held block
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, below = parts[1], parts[2]
                current_positions[block] = below
            elif predicate == 'on-table':
                block = parts[1]
                current_positions[block] = 'table'
            elif predicate == 'holding':
                currently_held = parts[1]
                current_positions[currently_held] = 'hand' # Special marker

        misplaced_count = 0
        # Iterate through blocks whose position is specified in the goal
        for block in self.goal_blocks:
            goal_pos = self.goal_positions.get(block) # 'table' or block_name

            # Find the current position of the block
            # Use .get() with a default like None in case a block from goal_blocks
            # is somehow not found in current_positions (e.g., state inconsistency)
            current_pos = current_positions.get(block)

            # If the block is in the goal structure (it should have a goal_pos)
            # and its current position doesn't match the goal position, it's misplaced.
            # This correctly handles blocks on other blocks, on the table, or in the hand.
            if goal_pos is not None and current_pos != goal_pos:
                 misplaced_count += 1

        # Component 2: Count unsatisfied 'clear' and 'arm-empty' goals
        additional_cost = 0

        # Check arm-empty goal
        if self.goal_arm_empty and "(arm-empty)" not in state:
             additional_cost += 1

        # Find blocks that are currently clear in the state
        state_clear_blocks = {get_parts(s)[1] for s in state if get_parts(s)[0] == 'clear'}

        # Check clear goals
        for block in self.goal_clear_blocks:
             if block not in state_clear_blocks:
                 additional_cost += 1

        # The heuristic is the sum of misplaced blocks (by position) and unsatisfied clear/arm-empty goals.
        return misplaced_count + additional_cost
