# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    # Basic check for fact format
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # This case should ideally not happen with standard PDDL fact strings
         # as represented in the state. Return empty list or handle error.
         return []
    return fact_str[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number of
    goal blocks that are not currently part of a correctly built stack, starting
    from the table up. A block is considered correctly stacked if it is in its
    goal position relative to the block (or table) below it, AND the block below
    it is also correctly stacked according to the goal structure.

    # Assumptions
    - The goal is defined by a set of (on X Y) and (on-table X) facts forming stacks.
    - The heuristic counts blocks that are part of the goal configuration but are
      not currently in their correct place within that configuration, considering
      the required stack order from the base (table) upwards.
    - Assumes standard Blocksworld actions and predicates.

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to identify the desired position
      for each block involved in the goal.
    - Stores the desired 'on' relationships in `self.goal_on_map` (mapping a block
      to the block it should be directly on top of).
    - Identifies the set of blocks that should be on the table in the goal state.
      These are the blocks mentioned in goal facts that are *not* supposed to be
      on top of any other block according to the goal (`self.goal_on_table`).
    - Collects all blocks involved in the goal into `self.goal_blocks`.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state (`node.state`) to determine the current position
       of each block. Store this information in `state_on_map` (mapping a block
       to the block it is currently directly on top of) and `state_on_table`
       (set of blocks currently on the table).
    2. Initialize a counter `incorrectly_stacked_goal_blocks` to 0.
    3. Define a recursive helper function `is_correctly_stacked_from_base(block)`
       (with memoization) that checks if a given `block` is in its final goal
       position relative to its base, and if that base is also correctly stacked,
       recursively down to the table.
       - If `block` should be on the table (`block in self.goal_on_table`), the check
         succeeds if `block` is currently in `state_on_table`.
       - If `block` should be on `Y` (`self.goal_on_map[block] == Y`), the check
         succeeds if `block` is currently on `Y` (`state_on_map.get(block) == Y`)
         AND `is_correctly_stacked_from_base(Y)` is true.
       - If `block` is not a goal block (i.e., not in `self.goal_blocks`), it doesn't
         break the correctness of a goal stack above it, so the check returns True
         for this branch of recursion.
    4. Iterate through each `block` in `self.goal_blocks`.
    5. For each goal `block`, call `is_correctly_stacked_from_base(block)`.
    6. If the function returns False, increment `incorrectly_stacked_goal_blocks`.
    7. The heuristic value is the final count of `incorrectly_stacked_goal_blocks`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Map to store the block a block should be on in the goal state.
        # e.g., { 'a': 'b', 'b': 'c' } if goal is (on a b) and (on b c)
        self.goal_on_map = {}
        # Set of blocks that are the bottom-most blocks of goal stacks (should be on table).
        # e.g., { 'c' } if goal is (on a b) and (on b c) and (on-table c)
        self.goal_on_table = set()
        # Set of all blocks that are part of the goal configuration.
        self.goal_blocks = set()

        # First pass to build goal_on_map and collect all goal blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                self.goal_on_map[block] = base
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif predicate == "on-table":
                 # Add to goal_blocks for now, will refine goal_on_table later
                 self.goal_blocks.add(parts[1])

        # Second pass to identify blocks that should be on the table.
        # These are goal blocks that are not bases for any other goal block.
        blocks_that_are_bases = set(self.goal_on_map.values())
        self.goal_on_table = {block for block in self.goal_blocks if block not in blocks_that_are_bases}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build current state representation
        state_on_map = {}
        state_on_table = set()
        # holding_block = None # Not needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                state_on_map[block] = base
            elif predicate == "on-table":
                block = parts[1]
                state_on_table.add(block)
            # elif predicate == "holding":
            #     holding_block = parts[1] # Not used in current logic

        # --- Recursive helper function to check if a block is correctly stacked from its base ---
        # Uses memoization to avoid redundant calculations for the same block in the same state.
        memo = {}
        def is_correctly_stacked_from_base(block):
            if block in memo:
                return memo[block]

            result = False
            # Case 1: Block should be on the table in the goal
            if block in self.goal_on_table:
                 # Check if it is currently on the table
                 result = block in state_on_table
            # Case 2: Block should be on another block in the goal
            elif block in self.goal_on_map:
                target_base = self.goal_on_map[block]
                # Check if it is currently on the target base
                if block in state_on_map and state_on_map[block] == target_base:
                    # If it's on the correct block, recursively check the block below it
                    result = is_correctly_stacked_from_base(target_base)
                else:
                    # It's not on the correct block
                    result = False
            # Case 3: Block is not a goal block. If we trace a stack down to a non-goal block,
            # that part of the stack below the goal block is irrelevant to the goal stack's correctness.
            # So, the stack is considered correct *down to this non-goal block*.
            else:
                 result = True

            memo[block] = result
            return result
        # --- End of recursive helper function ---


        # Count goal blocks that are not correctly stacked from their base
        incorrectly_stacked_count = 0
        for block in self.goal_blocks:
            if not is_correctly_stacked_from_base(block):
                incorrectly_stacked_count += 1

        return incorrectly_stacked_count
