from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()

    # Check for the standard PDDL fact format (e.g., "(predicate arg1 arg2)")
    if fact_str.startswith('(') and fact_str.endswith(')'):
        # Remove parentheses and split by whitespace
        return fact_str[1:-1].split()

    # Handle simple facts like "arm-empty" which might be represented without parens
    # in the state representation, although less common.
    # Or handle facts that are already just the predicate name string.
    if fact_str:
        # Assume it's a single-part fact if not in parentheses format
        return [fact_str]

    # Return empty list for empty or malformed input
    return []


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing costs for blocks that are not in their goal position. The cost
    includes moving the block itself and clearing any stacks currently on top
    of the block or its target location in the goal. It is designed to be
    non-admissible and guide a greedy best-first search.

    # Assumptions
    - Standard Blocksworld rules apply.
    - All blocks present in the initial state or goal are relevant and must
      be placed according to the goal predicates ((on X Y) or (on-table X)).
    - The goal defines a valid configuration of blocks (no cycles, unique
      parent for each block except those on the table).
    - The 'arm' is a special location for a held block. 'table' is a special
      location for blocks on the table.
    - The state representation is complete, meaning every block is either
      on the table, on another block, or held.

    # Heuristic Initialization
    - Parses the goal predicates ((on X Y) and (on-table X)) to determine the
      desired parent (the block below it, or 'table') for each block in the
      goal state. This mapping is stored in `self.goal_parent`. Blocks not
      mentioned as being 'on' another block or 'on-table' in the goal are
      assumed to not have a specific required parent in the goal (though
      standard Blocksworld problems usually specify locations for all blocks).
    - Identifies all unique blocks involved in the problem by collecting
      arguments from all relevant predicates in the initial state and goal.
      This set `self.all_blocks` is primarily for completeness; the heuristic
      calculation focuses on blocks with a defined goal parent.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state predicates ((on X Y), (on-table X), (holding X))
       to determine the current parent for each block (the block it's on,
       'table', or 'arm'). Store this in `current_parent`.
    2. Build a map `current_child_map` where `current_child_map[block_below] = block_on_top`
       for all `(on block_on_top block_below)` facts in the state. This map
       allows finding the block directly on top of another.
    3. Initialize the total heuristic cost `h = 0`.
    4. Define a helper function `count_on_top(block, child_map)` that
       iteratively counts the number of blocks currently stacked directly
       on top of the given block using the `child_map`.
    5. Iterate through all blocks that have a defined goal parent (i.e., blocks
       whose final position is specified in `self.goal_parent`):
        a. Get the block's current parent (`current_parent[block]`). We assume
           this block is present in `current_parent` in a valid state.
        b. Get the block's goal parent (`goal_loc`, which is the value from `self.goal_parent`).
        c. If the block is currently held (`current_loc == 'arm'`):
           If the block's goal is not to be held (`goal_loc != 'arm'`):
             - Add 1 to `h` (cost for a putdown or stack action).
             - If the goal parent is a block (not 'table' or 'arm'), add the
               number of blocks currently on top of the goal parent in the
               current state (`count_on_top(goal_loc, current_child_map)`) to `h`
               (cost to clear the target location).
        d. If the block is not held (`current_loc != 'arm'`) and
           its current parent is different from its goal parent (`current_loc != goal_loc`):
             - Add 2 to `h` (cost for a pickup and a putdown/stack action).
             - Add the number of blocks currently on top of this block in the
               current state (`count_on_top(block, current_child_map)`) to `h`
               (cost to clear this block).
             - If the goal parent is a block (not 'table' or 'arm'), add the
               number of blocks currently on top of the goal parent in the
               current state (`count_on_top(goal_loc, current_child_map)`) to `h`
               (cost to clear the target location).
    6. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal structure and identifying blocks.
        """
        super().__init__(task)

        # Map block to its goal parent (block below it, or 'table').
        self.goal_parent = {}
        # Set of all blocks in the problem.
        self.all_blocks = set() # Kept for completeness/potential future use

        # Parse goal predicates to build goal_parent map and find blocks
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block_on, block_below = parts[1], parts[2]
                self.goal_parent[block_on] = block_below
                self.all_blocks.add(block_on)
                self.all_blocks.add(block_below)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_parent[block] = 'table'
                self.all_blocks.add(block)
            # Collect blocks from other goal predicates like (clear X)
            elif predicate in ["clear", "holding"] and len(parts) > 1:
                 for arg in parts[1:]:
                     if arg not in ['table', 'arm-empty', 'arm']:
                         self.all_blocks.add(arg)


        # Also collect blocks from initial state facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate in ["on", "on-table", "clear", "holding"] and len(parts) > 1:
                 for arg in parts[1:]:
                     if arg not in ['table', 'arm-empty', 'arm']:
                         self.all_blocks.add(arg)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        current_parent = {}
        current_child_map = {} # Maps block_below -> block_on_top

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block_on, block_below = parts[1], parts[2]
                current_parent[block_on] = block_below
                current_child_map[block_below] = block_on
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_parent[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                block = parts[1]
                current_parent[block] = 'arm'

        def count_on_top(block, child_map):
            """
            Helper function to count the number of blocks currently stacked
            directly on top of the given block using the child_map.
            """
            count = 0
            temp_block = block
            # While there is a block directly on top of temp_block
            while temp_block in child_map:
                 block_above = child_map[temp_block]
                 count += 1
                 temp_block = block_above # Move up the stack
            return count


        total_cost = 0

        # Iterate through blocks that have a defined goal parent
        # (i.e., blocks whose final position is specified)
        for block, goal_loc in self.goal_parent.items():
            # Assuming valid states where every block is located somewhere
            # and is present in current_parent if it's in goal_parent.
            current_loc = current_parent[block]

            # Case 1: Block is currently held
            if current_loc == 'arm':
                # If the block should not be held in the goal
                if goal_loc != 'arm': # Goal location is 'table' or another block
                    total_cost += 1 # Cost for a putdown or stack action

                    # Cost to clear the goal location if it's a block
                    if goal_loc != 'table':
                        total_cost += count_on_top(goal_loc, current_child_map)

            # Case 2: Block is on table or another block
            elif current_loc != goal_loc: # Block is in the wrong place
                total_cost += 2 # Cost for a pickup and a putdown/stack action

                # Cost to clear the block itself
                total_cost += count_on_top(block, current_child_map)

                # Cost to clear the goal location if it's a block
                if goal_loc != 'table':
                    total_cost += count_on_top(goal_loc, current_child_map)

            # Case 3: Block is in the correct place (current_loc == goal_loc)
            # No cost added for this block itself being misplaced.
            # Blocks on top of it that shouldn't be there will be accounted for
            # when we iterate over those blocks (if they are also misplaced).

        return total_cost
