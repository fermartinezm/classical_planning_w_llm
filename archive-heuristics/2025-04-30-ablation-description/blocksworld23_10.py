from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct position within the goal stack configuration. A block is
    considered correctly placed if it is on the block specified by the
    goal (or on the table if the goal specifies) AND the block it is
    on is also correctly placed according to the goal.

    # Assumptions
    - The goal specifies a specific arrangement of blocks into stacks
      using 'on' and 'on-table' predicates.
    - Blocks not mentioned in goal 'on' or 'on-table' predicates are
      not considered by this heuristic in terms of being "correctly placed
      in a goal stack".

    # Heuristic Initialization
    - Parses the goal predicates to build the desired stack structure
      (`self.goal_state_map`), mapping each block to the block it should
      be directly on top of (or 'table').
    - Identifies the set of all blocks involved in the goal stacks
      (`self.goal_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state predicates to build the current stack
       structure (`current_state_map`), mapping each block to the block
       it is currently directly on top of (or 'table'). Blocks being held
       are not included in this map.
    2. Define a recursive helper function `is_correctly_placed(block)`
       that determines if a given block is in its correct goal position
       relative to the block below it, and if the block below it is also
       correctly placed. This function uses memoization to avoid redundant
       calculations and handle recursion.
       - Base case: If the block's goal is to be on the table, check if it's
         currently on the table.
       - Recursive step: If the block's goal is to be on block Y, check if
         it's currently on block Y AND if Y is correctly placed.
       - If a block is part of a goal stack but its own goal position is not
         defined in `self.goal_state_map` (e.g., it's only mentioned as an
         'under_block' but not as a block that is 'on' something or 'on-table'),
         it cannot be correctly placed within a goal stack context by this heuristic.
    3. Initialize a counter for misplaced blocks to 0.
    4. Iterate through each block in the set of goal blocks (`self.goal_blocks`).
    5. For each block, check if its goal position is defined (i.e., if it's a key
       in `self.goal_state_map`).
    6. If the goal position is defined, call `is_correctly_placed` for the block.
       If it returns False, increment the counter.
    7. If the goal position is *not* defined for a block in `self.goal_blocks`,
       it means this block is an intermediate or base block in a goal stack
       whose own position isn't specified in the goal. Such a block cannot be
       part of a correctly placed stack prefix, so it is counted as misplaced.
       Increment the counter in this case.
    8. The final value of the counter is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack configuration.
        """
        self.goals = task.goals

        # Build the goal stack map: block -> block_below_it (or 'table')
        self.goal_state_map = {}
        # Identify all blocks that are part of the goal stacks
        self.goal_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts if any
                continue
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                self.goal_state_map[block] = under_block
                self.goal_blocks.add(block)
                self.goal_blocks.add(under_block) # The block underneath is also part of the goal stack
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_state_map[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear', 'arm-empty', 'holding' goals for this stacking heuristic

        # Remove 'table' from goal_blocks if it was added
        self.goal_blocks.discard('table')


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Build the current stack map: block -> block_below_it (or 'table')
        current_state_map = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                block, under_block = parts[1], parts[2]
                current_state_map[block] = under_block
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_state_map[block] = 'table'
            # Ignore 'clear', 'arm-empty', 'holding' facts for the stacking map.
            # Blocks that are 'holding' will not appear in 'on' or 'on-table' facts,
            # so they won't be in current_state_map keys, which is correct.

        # Memoization cache for the recursive function
        correctly_placed_cache = {}

        def is_correctly_placed(block):
            """
            Checks if the block is in its correct goal position relative to
            the block below it, and if the block below it is also correctly placed.
            Assumes 'block' is a key in self.goal_state_map.
            """
            # If already computed, return cached result
            if block in correctly_placed_cache:
                return correctly_placed_cache[block]

            # This function should only be called for blocks that are keys in goal_state_map
            # (i.e., blocks whose goal position is explicitly defined).
            # assert block in self.goal_state_map, f"is_correctly_placed called for block {block} not in goal_state_map"


            goal_under = self.goal_state_map[block]
            current_under = current_state_map.get(block) # Use .get() in case block is held or not on anything

            result = False
            if goal_under == 'table':
                # Goal is on the table, check if it's currently on the table
                result = (current_under == 'table')
            else:
                # Goal is on another block, check if it's on the correct block
                # AND if the block below it is also correctly placed.
                # We must ensure goal_under is also in goal_state_map for the recursion
                # to represent a correctly placed stack prefix rooted at the table.
                if goal_under in self.goal_state_map:
                     result = (current_under == goal_under) and is_correctly_placed(goal_under)
                else:
                     # If the block below (goal_under) is part of the goal stack (in goal_blocks)
                     # but its own goal position isn't defined (not in goal_state_map),
                     # then it cannot be correctly placed, and thus 'block' cannot be either.
                     # If goal_under is not even in goal_blocks, it's an invalid goal structure.
                     # In either case, 'block' is not correctly placed relative to a valid goal stack.
                     result = False


            # Cache the result before returning
            correctly_placed_cache[block] = result
            return result

        # Count the number of blocks in goal stacks that are NOT correctly placed
        misplaced_count = 0
        # Iterate over all blocks involved in goal stacks
        for block in self.goal_blocks:
             # Check correctly placed status only for blocks whose goal position is defined
             # (i.e., blocks that are keys in goal_state_map).
             if block in self.goal_state_map:
                if not is_correctly_placed(block):
                    misplaced_count += 1
             else:
                 # If a block is in goal_blocks but not in goal_state_map, it means
                 # it's only mentioned as an 'under_block' in a goal 'on' predicate.
                 # Its own goal position isn't defined. It cannot be part of
                 # a correctly placed stack prefix rooted at the table.
                 # It is therefore misplaced relative to the goal structure.
                 misplaced_count += 1


        return misplaced_count
