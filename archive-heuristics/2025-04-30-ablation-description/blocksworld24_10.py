from heuristics.heuristic_base import Heuristic

def get_parts(fact_str):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    return fact_str[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct goal stack configuration. A block is considered "correctly stacked"
    if it is on the correct block (or the table) according to the goal, AND
    the block immediately below it *in the goal stack* is also correctly
    stacked (recursively), down to a block that is correctly on the table.
    The heuristic counts the total number of blocks that have a specified
    position in the goal state minus the number of these blocks that are
    correctly stacked.

    # Assumptions:
    - The goal specifies the desired position for each block that is part of
      a goal stack, either on another block or on the table.
    - Blocks not mentioned as the object of a goal `on` or `on-table` predicate
      are considered irrelevant to the goal stack structure being measured by
      this heuristic.
    - The heuristic counts blocks that are misplaced within the goal stack
      structure. It does not directly account for `clear` or `arm-empty`
      conditions, although achieving the correct stack structure typically
      requires satisfying these conditions along the way.

    # Heuristic Initialization
    - Parses the goal state to determine the desired block immediately below
      each block (or if it should be on the table) for all blocks involved
      in goal `on` or `on-table` predicates. This information is stored
      in the `_all_goal_on_below` dictionary.
    - Identifies all blocks that are mentioned as the object of a goal `on`
      or `on-table` predicate. These are the blocks whose positions are
      explicitly constrained by the goal and will be counted by the heuristic.
      This set is stored in `goal_blocks_with_pos`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the block immediately below each
       block (or if it is on the table or being held). Store this information
       by creating a map `current_on_map` (mapping block_below -> block_on_top)
       and a set `current_on_table` (set of blocks on the table). Identify the
       `currently_held` block if any (though the held status is implicitly
       handled by absence from `current_on_map` and `current_on_table`).
    2. Initialize a set `correctly_stacked_blocks` to store blocks that are
       determined to be in their correct goal stack position.
    3. Add blocks to `correctly_stacked_blocks` if their goal is to be on the
       table (`(on-table X)` is a goal fact, checked using `_all_goal_on_below`)
       AND they are currently on the table (`(on-table X)` is a fact in the
       current state). These blocks form the correct bases of goal stacks.
    4. Create a set of all goal `(on X Y)` facts based on the `_all_goal_on_below` mapping.
    5. Iteratively expand the set of `correctly_stacked_blocks`:
       - Loop until no new blocks are added to `correctly_stacked_blocks` in an iteration.
       - In each iteration, check each goal `(on X Y)` fact from the set created in step 4.
       - If `(on X Y)` is true in the current state (i.e., Y is in `current_on_map`
         and `current_on_map[Y]` is X) AND block Y is already in the
         `correctly_stacked_blocks` set, then block X is also correctly
         stacked on top of a correctly stacked block. Add X to a temporary set
         of newly stacked blocks.
       - Add all blocks from the temporary set of newly stacked blocks to the
         main `correctly_stacked_blocks` set.
    6. The heuristic value is the total number of blocks whose position is
       explicitly constrained by the goal (i.e., blocks in `self.goal_blocks_with_pos`)
       minus the number of these blocks that are found in the `correctly_stacked_blocks` set.
       This counts how many goal-constrained blocks are *not* in their correct
       goal stack configuration.
    7. If the full goal state (all goal facts, including `clear` and `arm-empty`)
       is met, the heuristic value is 0. This check is performed first.
       Otherwise, the calculated count from step 6 is returned.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal positions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Map block to the block it should be on top of in the goal state,
        # or 'table' if it should be on the table. This includes all blocks
        # involved in goal 'on' or 'on-table' predicates.
        self._all_goal_on_below = {}
        # Keep track of all blocks that are the subject of a goal 'on' or 'on-table' predicate.
        # These are the blocks whose positions are explicitly constrained by the goal
        # and will be counted by the heuristic if misplaced.
        self.goal_blocks_with_pos = set()


        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                self._all_goal_on_below[block] = below
                self.goal_blocks_with_pos.add(block)
            elif parts[0] == 'on-table':
                block = parts[1]
                self._all_goal_on_below[block] = 'table'
                self.goal_blocks_with_pos.add(block)
            # Ignore 'clear' and 'arm-empty' goals for the core stack calculation


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the goal is reached. If so, heuristic must be 0.
        if self.goals <= state:
             return 0

        # Map block_below -> block_on_top
        current_on_map = {}
        # Set of blocks currently on the table
        current_on_table = set()
        # The block currently held by the arm (not strictly needed for this heuristic logic)
        # currently_held = None

        # Parse the current state to find current positions
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, below = parts[1], parts[2]
                current_on_map[below] = block
            elif parts[0] == 'on-table':
                block = parts[1]
                current_on_table.add(block)
            # elif parts[0] == 'holding':
            #     currently_held = parts[1]
            # Ignore 'clear', 'arm-empty', 'holding' (except indirectly)

        # Set to store blocks that are correctly stacked according to the goal
        correctly_stacked_blocks = set()

        # Add blocks whose goal is on-table and are currently on-table
        # We iterate through all blocks that have *any* goal position specified
        # in _all_goal_on_below to find the bases of goal stacks.
        for block, goal_pos_below in self._all_goal_on_below.items():
            if goal_pos_below == 'table' and block in current_on_table:
                correctly_stacked_blocks.add(block)

        # Get goal (on X Y) facts for all blocks involved in goal stacks
        all_goal_on_facts = {
            (block, self._all_goal_on_below[block])
            for block in self._all_goal_on_below
            if self._all_goal_on_below[block] != 'table'
        }

        # Iteratively add blocks that are correctly stacked on top of already correctly stacked blocks
        changed = True
        while changed:
            changed = False
            newly_stacked = set()
            # Iterate over all goal (on X Y) facts
            for block, below in all_goal_on_facts:
                # Check if (on block below) is true in the current state
                is_on_in_state = (below in current_on_map and current_on_map[below] == block)

                # If block is currently on the correct block 'below', AND 'below' is correctly stacked
                if is_on_in_state and below in correctly_stacked_blocks:
                    # If 'block' is not already marked as correctly stacked
                    if block not in correctly_stacked_blocks:
                        newly_stacked.add(block)
                        changed = True # We added a new block, so loop again

            correctly_stacked_blocks.update(newly_stacked)

        # The heuristic is the number of blocks with a specified goal position
        # (those in self.goal_blocks_with_pos) that are NOT correctly stacked.
        # We only count blocks whose *own* position is constrained by the goal.
        misplaced_count = len(self.goal_blocks_with_pos) - len(correctly_stacked_blocks.intersection(self.goal_blocks_with_pos))

        # Ensure heuristic is non-negative (should be, but for safety)
        return max(0, misplaced_count)
