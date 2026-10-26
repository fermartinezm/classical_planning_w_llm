# from heuristics.heuristic_base import Heuristic # Assuming this is provided

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL states are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# The actual heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of blocks that are not in their correct goal position
    relative to their support, or have an incorrect block stacked directly on top
    according to the goal. Each such block contributes 1 to the heuristic value.

    # Assumptions
    - The goal specifies the desired final configuration of blocks using `on` and `on-table` predicates.
    - The heuristic focuses on achieving the correct relative positions between blocks and the table.
    - The heuristic does not explicitly model the arm state or clearance requirements,
      assuming that achieving the correct block positions is the primary cost driver.
    - All blocks involved in the problem are mentioned in the initial state or goal state.

    # Heuristic Initialization
    - Parses the goal facts to build the target configuration:
        - `goal_config`: Maps each block that needs a specific support in the goal
          to its required support (another block or 'table').
        - `goal_on_top`: Maps each block (that is a support in the goal) to the
          block that should be directly on top of it in the goal.
    - Collects the set of all blocks involved in the problem by examining initial
      and goal state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine the current position of every block:
       - `current_support`: Maps each block to its current support ('table',
         another block, or 'arm' if held). Initialize all known blocks to have
         no support initially, then populate based on state facts.
       - `current_on_top`: Maps each block (that is a support) to the block
         currently directly on top of it.
       - Identify the block currently being held, if any.
    2. Initialize the heuristic cost to 0.
    3. Iterate through each block that is part of the goal configuration
       (i.e., appears as a key in `goal_config`). These are the blocks whose
       position relative to their support is explicitly specified in the goal.
    4. For each block `B` from step 3:
       - Determine its required goal support (`goal_support = self.goal_config[B]`).
       - Determine its current support (`current_support_val = current_support.get(B)`).
       - Check if the current support matches the goal support.
       - If `current_support_val != goal_support`, increment the heuristic cost by 1.
       - If `current_support_val == goal_support` (the block is on the correct support),
         check if the block directly on top of `B` in the current state
         (`current_block_on_top = current_on_top.get(B)`) matches the block that
         should be directly on top of `B` in the goal (`goal_block_on_top = self.goal_on_top.get(B)`).
       - If `current_support_val == goal_support` but `current_block_on_top != goal_block_on_top`
         (this covers cases where a block should be clear but isn't, or shouldn't
         be clear but is, or has the wrong block on top), increment the heuristic cost by 1.
    5. The total heuristic value is the sum of costs accumulated in step 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and all blocks.
        """
        self.goals = task.goals

        # Build goal configuration mappings
        self.goal_config = {}  # block -> support (block or 'table')
        self.goal_on_top = {}  # support (block) -> block_on_top

        # Collect all blocks mentioned in initial state and goals
        self.all_blocks = set()

        # Helper to add arguments from a fact to the set of all blocks
        def add_args_to_blocks(fact_parts):
             if not fact_parts: return
             # Add all arguments that are not standard predicates or keywords
             for part in fact_parts[1:]:
                 if part not in ['table', 'arm-empty', 'clear', 'holding', 'on', 'on-table']:
                      self.all_blocks.add(part)

        for goal in self.goals:
            parts = get_parts(goal)
            add_args_to_blocks(parts)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                self.goal_config[obj] = underob
                self.goal_on_top[underob] = obj
            elif predicate == "on-table":
                obj = parts[1]
                self.goal_config[obj] = 'table'
            # Ignore (clear ?) and (arm-empty) goals for this heuristic's structure

        for fact in task.initial_state:
             parts = get_parts(fact)
             add_args_to_blocks(parts)
             # We don't need to parse initial state structure here, just collect block names


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Parse current state
        current_support = {} # block -> support (block, 'table', or 'arm')
        current_on_top = {}  # support (block) -> block_on_top
        current_holding = None

        # Initialize support for all blocks to None (should be overwritten by state facts)
        # This is mainly defensive; valid states should place all blocks.
        for block in self.all_blocks:
             current_support[block] = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on":
                obj, underob = parts[1], parts[2]
                current_support[obj] = underob
                current_on_top[underob] = obj
            elif predicate == "on-table":
                obj = parts[1]
                current_support[obj] = 'table'
            elif predicate == "holding":
                obj = parts[1]
                current_support[obj] = 'arm' # Special support type for held blocks
                current_holding = obj
            # Ignore (clear ?) and (arm-empty) state facts for this heuristic's structure


        total_cost = 0

        # Iterate through blocks that have a specific goal position defined by on/on-table
        # These are the blocks whose placement we care about for the goal stacks.
        for block in self.goal_config.keys():
            goal_support = self.goal_config[block]
            current_support_val = current_support.get(block) # Use .get() in case block wasn't found in state parsing

            # Check if the block is on the correct support
            if current_support_val != goal_support:
                total_cost += 1
            else:
                # Support is correct, now check the block on top
                goal_block_on_top = self.goal_on_top.get(block)
                current_block_on_top = current_on_top.get(block)

                # Check if the block on top is correct
                # This condition is true if current_block_on_top is not the same as goal_block_on_top
                # (This covers: should be clear but isn't, shouldn't be clear but is, wrong block on top)
                if current_block_on_top != goal_block_on_top:
                     total_cost += 1

        return total_cost
