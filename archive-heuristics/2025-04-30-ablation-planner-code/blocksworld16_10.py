# Assume heuristics.heuristic_base exists and defines a Heuristic class
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided externally, you might need a mock definition
# for testing purposes, but the final code assumes it's available.
# Example Mock Heuristic base class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         pass
#     def __call__(self, node):
#         pass

# No external libraries like fnmatch are strictly needed for this specific heuristic logic.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Assuming valid PDDL strings, but returning empty list for safety
         return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of goal conditions that are not met,
    prioritizing the correct placement of blocks within goal stacks from the
    bottom up. It counts:
    1. Blocks that are part of a goal stack but are not in their correct position
       relative to the block directly below them (or the table), considering
       whether the block below is itself in its correct position.
    2. Top blocks in goal stacks that are not clear.
    3. A small penalty if the arm is not empty and the goal requires it to be empty.

    A block B is considered "in place" relative to the goal stack structure if:
    - It is supposed to be on the table (goal `(on-table B)`) AND it is currently on the table (`(on-table B)` is true in the state).
    - OR it is supposed to be on block U (goal `(on B U)`) AND it is currently on block U (`(on B U)` is true in the state) AND block U is also "in place" relative to the goal stack structure below it.

    # Assumptions
    - The goal state consists of one or more stacks of blocks, possibly with some blocks on the table.
    - Goal conditions primarily define the `on` and `on-table` relationships for the desired stacks, and `clear` for the top blocks.
    - The goal defines complete stacks down to the table for all blocks involved in `on` predicates.

    # Heuristic Initialization
    - Parse the goal conditions (`task.goals`).
    - Build the target stack structure:
        - `goal_on_map`: maps block -> block_below for each goal `(on block block_below)`.
        - `goal_on_table_set`: set of blocks for which the goal is `(on-table block)`.
    - Identify blocks that are designated as the top of a goal stack (those with a `(clear block)` goal that are not specified as being below any other block in an `on` goal).
    - Identify the set of all blocks whose position is explicitly defined in the goal structure (`blocks_to_check`).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the heuristic value `h` to 0.
    2. For each block `B` in the set `blocks_to_check` (blocks whose position is explicitly defined in the goal):
       - Check if `B` is "in place" using a recursive helper function `is_in_place(B, state, goal_on_map, goal_on_table_set, blocks_to_check, memo)`.
       - The `is_in_place` function determines correctness based on the recursive definition (on table OR on correct block AND block below is in place). It uses memoization (`memo`) to store results for blocks already checked during the recursion, preventing redundant computation and handling potential cycles (though cycles shouldn't occur in valid Blocksworld goals). It also checks if the block below is itself part of the defined goal structure.
       - If `is_in_place(B, state, ...)` returns False, increment `h`. This counts blocks that are part of the desired stacks but are not currently in their correct final position relative to the structure below them.
    4. Identify blocks that are supposed to be at the top of a goal stack. These are blocks `B` for which `(clear B)` is a goal AND `B` is not specified as being below any other block in any `(on X B)` goal. This set (`goal_top_blocks`) is precomputed in `__init__`.
    5. For each block `B` in `goal_top_blocks`:
       - If `(clear B)` is not true in the current state, increment `h`. This penalizes having extra blocks on top of a completed or partially completed goal stack.
    6. If `(arm-empty)` is a goal condition and `(arm-empty)` is not true in the current state, increment `h` by 1. This adds a small penalty for needing to clear the arm.
    7. Return the final value of `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the goal stack structure and top blocks.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # static_facts = task.static # Not used

        # Build the goal stack structure
        self.goal_on_map = {} # Maps block -> block_below for (on block block_below) goals
        self.goal_on_table_set = set() # Set of blocks that should be on-table

        # Identify blocks that are specified as being below another block in an 'on' goal
        blocks_that_are_below = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, block_below = parts[1], parts[2]
                self.goal_on_map[block] = block_below
                blocks_that_are_below.add(block_below)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_on_table_set.add(block)
            # Ignore other predicates like 'clear' or 'arm-empty' for building the stack structure maps

        # Identify blocks that are designated as the top of a goal stack.
        # These are blocks B for which (clear B) is a goal AND B is NOT a block_below
        # for any other block in an (on X B) goal.
        self.goal_top_blocks = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             if predicate == "clear" and len(parts) == 2:
                 block = parts[1]
                 if block not in blocks_that_are_below:
                      self.goal_top_blocks.add(block)

        # The set of blocks whose position is explicitly defined in the goal structure
        # (i.e., those appearing as the first argument in an (on ...) goal or in an (on-table ...) goal).
        self.blocks_to_check = set(self.goal_on_map.keys()) | self.goal_on_table_set


    def is_in_place(self, block, state, goal_on_map, goal_on_table_set, blocks_to_check, memo):
        """
        Helper function to recursively check if a block is in its correct goal position
        relative to the stack structure below it. Uses memoization.
        """
        if block in memo:
            return memo[block]

        result = False # Default: not in place

        if block in goal_on_table_set:
            # Goal: block is on the table
            result = f"(on-table {block})" in state
        elif block in goal_on_map:
            # Goal: block is on block_below
            block_below = goal_on_map[block]
            # Check if block is currently on block_below
            if f"(on {block} {block_below})" in state:
                 # Check if the block below is in place.
                 # It must be part of the goal structure below this point.
                 if block_below in blocks_to_check: # Use the precomputed set
                     result = self.is_in_place(block_below, state, goal_on_map, goal_on_table_set, blocks_to_check, memo)
                 else:
                     # block_below is not part of the defined goal structure below this point
                     # This indicates an issue with the goal definition or assumption.
                     # Treat as not in place below.
                     result = False
            else:
                 result = False # block is not currently on block_below
        # else: The block is not explicitly placed in the goal structure (not a key in goal_on_map, not in goal_on_table_set).
        # It should not be checked by the main loop if blocks_to_check is defined correctly.
        # If somehow called, default result=False is appropriate, but shouldn't happen.

        memo[block] = result
        return result


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Counts blocks that are part of the goal stack structure but are not "in place",
        plus penalties for unmet clear goals on top blocks and unmet arm-empty goal.
        """
        state = node.state
        memo = {} # Memoization dictionary for is_in_place

        h = 0

        # 1. Count blocks in goal stacks that are not in place
        for block in self.blocks_to_check:
            # Pass blocks_to_check to the helper
            if not self.is_in_place(block, state, self.goal_on_map, self.goal_on_table_set, self.blocks_to_check, memo):
                h += 1

        # 2. Count top blocks that are not clear
        for block in self.goal_top_blocks:
             if f"(clear {block})" not in state:
                  h += 1

        # 3. Add penalty if arm-empty is a goal and arm is not empty
        if "(arm-empty)" in self.goals and "(arm-empty)" not in state:
             h += 1

        return h
