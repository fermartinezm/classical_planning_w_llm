# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle unexpected format - log warning or raise error if necessary
         # For this problem, assuming valid PDDL fact strings.
         return [] # Return empty list for malformed facts

    # Remove parentheses and split by whitespace
    return fact_str[1:-1].split()


class blocksworldHeuristic: # Inherit from Heuristic if base class is provided
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. The number of blocks that are part of the goal configuration but are not
       in their correct position relative to the block immediately below them
       in the goal stack, considering the goal stacks from the bottom up.
    2. The number of `(clear X)` goal facts that are not satisfied, specifically
       for blocks X that are *not* part of the main goal stack structures
       defined by `(on ...)` and `(on-table ...)` goal facts.

    # Assumptions
    - The goal state consists of specific stacks of blocks or blocks on the table,
      defined by `(on ...)` and `(on-table ...)` facts.
    - Additional `(clear ...)` goal facts might exist for blocks not involved
      in the main goal stacks.
    - Blocks not mentioned in any goal fact do not need to be in a specific location
      or state (like being clear).
    - The heuristic assumes that to place a block correctly in a stack, all blocks
      above it must first be moved, and the block below it must be correctly placed.

    # Heuristic Initialization
    - Parses the goal facts to build the desired goal configuration (`goal_config`),
      mapping each block to the block it should be on, or 'table'.
    - Identifies the set of all blocks that are part of the goal stack structures
      (`goal_blocks`) based on `(on ...)` and `(on-table ...)` goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Initialization (`__init__`)**:
       - Parse `task.goals`.
       - Build `self.goal_config`: a dictionary mapping a block `X` to `Y` if `(on X Y)` is a goal, or to `'table'` if `(on-table X)` is a goal.
       - Build `self.goal_blocks`: a set containing all blocks that appear in the `(on ...)` or `(on-table ...)` goal facts. These are the blocks involved in the goal stack structures.
    2. **Heuristic Calculation (`__call__`)**:
       - Parse the current state facts (`node.state`) to create `current_config`: a dictionary mapping a block `X` to `Y` if `(on X Y)` is true, or to `'table'` if `(on-table X)` is true. Blocks that are held are not keys in this dictionary.
       - Identify `blocks_that_are_below_something`: a set of blocks `Y` such that `(on X Y)` is true in the current state for some `X`. This helps quickly check if a block is clear.
       - Define a recursive helper function `is_correctly_placed(block, current_cfg, goal_cfg, memo)`:
         - This function checks if `block` is in its correct goal position relative to the block below it, AND if the block below it is also correctly placed (recursively).
         - It uses the `memo` dictionary for memoization to avoid redundant calculations.
         - Base case: If `block` should be on the table (`goal_cfg.get(block) == 'table'`), it's correctly placed if `current_cfg.get(block) == 'table'`.
         - Recursive step: If `block` should be on `block_below` (`goal_cfg.get(block) == block_below`), it's correctly placed if `current_cfg.get(block) == block_below` AND `is_correctly_placed(block_below, current_cfg, goal_cfg, memo)`.
         - If `block` is in `self.goal_blocks` but not found in `current_cfg` (meaning it's held or in an incorrect stack position), it's not correctly placed.
         - If `block` is not in `self.goal_blocks`, it's not considered for this part of the heuristic.
       - Initialize an empty memoization dictionary `memo = {}`.
       - Count `correctly_placed_count`: Iterate through each `block` in `self.goal_blocks` and call `is_correctly_placed(block, current_config, self.goal_config, memo)`. Increment the count if it returns `True`.
       - Calculate the first component of the heuristic: `h1 = len(self.goal_blocks) - correctly_placed_count`. This is the number of blocks in the goal stacks that are not part of a correctly built segment from the bottom up.
       - Count `unsatisfied_clear_goals_not_in_goal_blocks`: Iterate through `task.goals`. If a goal is `(clear X)` and `X` is *not* in `self.goal_blocks`, check if `(clear X)` is true in the current state using the `blocks_that_are_below_something` set. If not, increment the count.
       - Calculate the second component of the heuristic: `h2 = unsatisfied_clear_goals_not_in_goal_blocks`.
       - The total heuristic value is `h1 + h2`.
       - This heuristic is 0 if and only if the state is the goal state, and is admissible.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal configuration."""
        self.goals = task.goals

        # 1. Parse goal facts to build goal_config and goal_blocks
        self.goal_config = {}
        blocks_in_on_goals = set()
        blocks_on_table_goals = set()

        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue

             predicate = parts[0]
             if predicate == "on":
                 if len(parts) == 3:
                     block_x, block_y = parts[1], parts[2]
                     self.goal_config[block_x] = block_y
                     blocks_in_on_goals.add(block_x)
                     blocks_in_on_goals.add(block_y)
             elif predicate == "on-table":
                 if len(parts) == 2:
                     block_x = parts[1]
                     self.goal_config[block_x] = 'table'
                     blocks_on_table_goals.add(block_x)

        # All blocks that are part of the goal stacks or on the table in the goal
        self.goal_blocks = blocks_in_on_goals | blocks_on_table_goals

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 2. Parse current state facts to build current_config and blocks_that_are_below_something
        current_config = {}
        blocks_that_are_below_something = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                if len(parts) == 3:
                    block_x, block_y = parts[1], parts[2]
                    current_config[block_x] = block_y
                    blocks_that_are_below_something.add(block_y)
            elif predicate == "on-table":
                if len(parts) == 2:
                    block_x = parts[1]
                    current_config[block_x] = 'table'
            # Ignore 'holding' facts - held blocks are not in current_config

        # Helper to check if a block is clear in the current state
        def is_clear_in_state(block, state_facts_below):
             return block not in state_facts_below

        # 3. Define recursive helper with memoization
        memo = {}
        def is_correctly_placed(block, current_cfg, goal_cfg, memo):
            # If block is not part of the goal configuration, it doesn't contribute
            # to the count of correctly placed blocks *in the goal structure*.
            if block not in self.goal_blocks:
                 return False # Should not be called for blocks outside goal_blocks

            if block in memo:
                return memo[block]

            goal_pos = goal_cfg.get(block)
            current_pos = current_cfg.get(block) # Will be None if block is held or not on table/stack

            # 5. Check base case and recursive step
            if current_pos is None or current_pos != goal_pos:
                # Block is held, or not in the correct position relative to the one below
                memo[block] = False
                return False

            if goal_pos == 'table':
                # Block is on table in both goal and current state
                memo[block] = True
                return True
            else:
                # Block is on block_below in both goal and current state
                block_below = goal_pos
                # Recursively check if the block below is correctly placed
                # Ensure block_below is in goal_config before recursing (should be true for valid goals)
                if block_below not in goal_cfg:
                     # This indicates an issue with goal parsing or PDDL structure
                     # For robustness, assume this stack segment is not correctly placed
                     memo[block] = False
                     return False

                result = is_correctly_placed(block_below, current_cfg, goal_cfg, memo)
                memo[block] = result
                return result

        # 8. Count correctly placed blocks in goal_blocks
        correctly_placed_count = 0
        memo = {} # Reset memoization cache for each state
        for block in self.goal_blocks:
            if is_correctly_placed(block, current_config, self.goal_config, memo):
                correctly_placed_count += 1

        # Calculate h1: Number of blocks in goal_blocks that are NOT correctly placed
        h1 = len(self.goal_blocks) - correctly_placed_count

        # 9. Count unsatisfied (clear X) goals where X is not in goal_blocks
        unsatisfied_clear_goals_not_in_goal_blocks = 0
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == "clear" and len(parts) == 2:
                block_x = parts[1]
                if block_x not in self.goal_blocks:
                    # Check if (clear block_x) is true in the current state
                    if not is_clear_in_state(block_x, blocks_that_are_below_something):
                        unsatisfied_clear_goals_not_in_goal_blocks += 1

        # Calculate h2: Number of unsatisfied (clear X) goals for blocks not in goal_blocks
        h2 = unsatisfied_clear_goals_not_in_goal_blocks

        # 10. Total heuristic value
        heuristic_value = h1 + h2

        return heuristic_value
