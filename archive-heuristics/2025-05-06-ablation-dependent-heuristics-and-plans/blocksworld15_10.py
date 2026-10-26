from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    # Removes leading/trailing parentheses and splits by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of structural misalignments in the
    current state compared to the goal state. It counts two types of
    discrepancies for each block: whether it is on the correct base (the
    block below it or the table), and whether the correct block is on top
    of it (or if it should be clear). Each such misalignment contributes 1
    to the heuristic value.

    # Assumptions:
    - Every block that is part of the goal configuration has a specified
      position, either on another block (`on X Y`) or on the table (`on-table X`).
    - The goal state implies a specific stack structure for all relevant blocks.
    - The heuristic counts local structural errors: wrong base and wrong block on top.
    - The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Heuristic Initialization
    - Extract all unique blocks involved in the problem instance by looking at
      the arguments of relevant predicates in both the initial and goal states.
    - Parse the goal state to precompute the desired base (block or 'table')
      for each block. This is stored in `self.goal_below_map`.
    - Parse the goal state to precompute the block that should be directly on
      top of each block in the goal state (or None if it should be clear on top).
      This is stored in `self.goal_on_top_map`. If a block should be clear on top
      in the goal, it will not be a key in this map.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. The `__init__` method identifies all blocks and precomputes the goal
       configuration mappings (`self.goal_below_map`, `self.goal_on_top_map`).
    2. The `__call__` method receives the current state (`node.state`).
    3. Parse the current state to create temporary mappings representing the
       current configuration:
       - `current_below_map`: Maps each block to the block (or 'table' or 'arm')
         it is currently directly on top of.
       - `current_on_top_map`: Maps each block to the block currently directly
         on top of it (or None if clear).
       - Identify the block currently being held, if any.
    4. Initialize the heuristic value `h = 0`.
    5. Iterate through each unique block identified during initialization:
       a. Determine the block's current base using `current_below_map.get(block)`.
       b. Determine the block's desired base using `self.goal_below_map.get(block)`.
       c. If the block has a specified desired base (`goal_b is not None`) and
          its current base is different from the desired base (`current_b != goal_b`),
          increment `h`. This counts blocks that are not on the correct layer
          of their goal stack or not on the table when they should be.
       d. Determine the block currently on top using `current_on_top_map.get(block)`.
       e. Determine the block that should be on top using `self.goal_on_top_map.get(block)`.
       f. If the block currently on top is different from the block that should be
          on top in the goal (`block_on_top_current != block_on_top_goal`), increment `h`.
          This counts blocks that are incorrectly topped. This covers cases where:
          - The wrong block is on top.
          - Something is on top when the block should be clear in the goal.
          - Nothing is on top when something specific should be there in the goal.
    6. The final value of `h` is the sum of all such misalignments across all blocks.
       This value is returned as the heuristic estimate.
    7. The heuristic value is 0 if and only if all blocks are on their correct bases
       and have the correct blocks (or nothing) on top, matching the goal stack structure.
       For standard Blocksworld goals, this implies the goal state is reached.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration and blocks."""
        super().__init__(task) # Call the base class constructor if needed, though Heuristic base is simple

        self.goals = task.goals

        # Extract all blocks from initial state and goal state
        self.blocks = set()
        # Collect all objects mentioned in relevant predicates in initial and goal states
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             # Consider predicates that involve blocks as arguments
             if parts and parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Add all arguments except 'table' which is a location, not a block object
                 for obj in parts[1:]:
                     if obj != 'table':
                         self.blocks.add(obj)

        # Precompute goal base and goal block on top for quick lookup
        self.goal_below_map = {} # Maps block -> block_below_it_in_goal or 'table'
        self.goal_on_top_map = {} # Maps block_below -> block_on_top_in_goal (None if block_below should be clear)

        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                self.goal_below_map[block_on] = block_below
                self.goal_on_top_map[block_below] = block_on # block_on is on top of block_below
            elif parts and parts[0] == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                self.goal_below_map[block_on_table] = 'table'
            # Note: (clear X) goals are implicitly handled. If (clear X) is a goal,
            # X will not appear as block_below in any (on Y X) goal, so
            # self.goal_on_top_map.get(X) will correctly return None.

        # It's assumed that all blocks relevant to the goal are covered by
        # 'on' or 'on-table' goal facts, thus present in self.goal_below_map.
        # Blocks in the initial state not in any goal position are still included
        # in self.blocks and will be evaluated; their goal_b will be None.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings
        h = 0

        # Build current state maps for quick lookup
        current_below_map = {} # Maps block -> block_below_it_in_state or 'table' or 'arm'
        current_on_top_map = {} # Maps block_below -> block_on_top_in_state (None if block_below is clear)
        # current_holding is not strictly needed for the heuristic calculation logic below,
        # as the 'holding' fact affects current_below_map.

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                current_below_map[block_on] = block_below
                current_on_top_map[block_below] = block_on # block_on is on top of block_below
            elif parts and parts[0] == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                current_below_map[block_on_table] = 'table'
            elif parts and parts[0] == 'holding' and len(parts) == 2:
                 held_block = parts[1]
                 current_below_map[held_block] = 'arm' # Block is held

        # Heuristic calculation
        for block in self.blocks:
            # Get current and goal base for the block
            current_b = current_below_map.get(block) # None if block is not in state facts (shouldn't happen for blocks in self.blocks)
            goal_b = self.goal_below_map.get(block) # None if block has no specified base in goal

            # Part 1: Count blocks on the wrong base
            # We only add cost if the block has a specified goal base and it's not there.
            # If goal_b is None, it means the block's base isn't specified in the goal,
            # so we don't count it as a base mismatch.
            if goal_b is not None and current_b != goal_b:
                 h += 1

            # Part 2: Count blocks where the block on top is wrong
            # This applies to blocks that are *not* the topmost in their goal stack,
            # or blocks that *are* topmost but have something on them in the state.
            # Find the block currently on top of 'block'
            block_on_top_current = current_on_top_map.get(block) # None if 'block' is clear in state
            # Find the block that should be on top of 'block' in the goal
            block_on_top_goal = self.goal_on_top_map.get(block) # None if 'block' should be clear in goal

            # If the block currently on top is different from the block that should be on top in the goal,
            # it's a misalignment. This covers:
            # - Wrong block on top (current_t is X, goal_t is Y, X != Y)
            # - Something on top when it should be clear (current_t is X, goal_t is None)
            # - Nothing on top when something specific should be there (current_t is None, goal_t is Y)
            if block_on_top_current != block_on_top_goal:
                 h += 1

        # The heuristic is 0 iff for every block, its base is correct AND its top is correct.
        # This corresponds exactly to the goal stack configuration being achieved.
        # Assuming standard blocksworld goals that specify positions for all blocks
        # and require the arm to be empty implicitly by requiring blocks to be on table/other blocks,
        # this implies the goal state is reached.

        return h
