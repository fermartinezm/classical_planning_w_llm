from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number of blocks
    that are part of the desired goal stacks but are not currently in their correct
    position relative to the block directly below them within a correctly formed
    portion of a goal stack, starting from the table.

    # Assumptions
    - The goal state defines one or more desired stacks of blocks on the table,
      specified by `(on x y)` and `(on-table z)` predicates.
    - The heuristic focuses on building these goal stacks from the bottom up.
    - The heuristic does not explicitly count actions, but rather the number of
      misplaced blocks within the goal structure that need to be moved or rearranged.
    - The heuristic assumes that any block not in its correct goal stack position
      needs at least some work (like moving it or clearing blocks on top of it)
      to eventually get it into place.

    # Heuristic Initialization
    - Parses the goal predicates to identify the desired block directly under each
      block in the goal stacks (`goal_under`) and the blocks that should be
      directly on the table (`goal_on_table`).
    - Identifies the set of all blocks that are part of the goal stack structure
      (`goal_blocks`). Blocks mentioned in `(clear x)` or `(arm-empty)` goals
      are not considered part of the core stack structure for this heuristic's
      counting mechanism, although achieving them is necessary for plan execution.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the desired position for each block involved in the goal stacks
       by parsing the goal predicates `(on x y)` and `(on-table z)`. Store the
       desired block underneath `x` in `goal_under[x] = y` and collect blocks
       that should be on the table in `goal_on_table`. Collect all blocks that
       appear as the first argument of an `on` goal or the argument of an
       `on-table` goal into `goal_blocks`.
    2. Identify the current position for each block in the given state by parsing
       the state predicates `(on x y)` and `(on-table z)`. Store this in
       `current_under[x] = y` and `current_on_table`.
    3. Initialize a set `in_goal_stack_position` to be empty. This set will store
       blocks that are currently in their correct position *relative to the block
       below them in the goal stack*, AND the block below them is also in its
       correct position (recursively down to the table).
    4. Add all blocks `b` to `in_goal_stack_position` if `(on-table b)` is a goal
       predicate AND `(on-table b)` is true in the current state. These are the
       correctly placed bases of the goal stacks.
    5. Repeatedly iterate and add blocks `x` to `in_goal_stack_position` if
       `(on x y)` is a goal predicate AND `(on x y)` is true in the current state
       AND `y` is already in `in_goal_stack_position`. This process builds the
       correctly placed stacks upwards from the base. Continue until no new blocks
       are added in an iteration.
    6. The heuristic value is the number of blocks in `goal_blocks` that are NOT
       in the final `in_goal_stack_position` set. This counts how many blocks
       that belong in a goal stack are not currently part of a correctly built
       segment of that stack. A value of 0 indicates that all blocks intended
       to be in specific stack positions are correctly placed relative to the
       goal stack structure from the bottom up. This state is the goal state
       (with respect to `on` and `on-table` predicates).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack structure.
        """
        self.goals = task.goals
        self.goal_under = {}
        self.goal_on_table = set()
        self.goal_blocks = set()

        # Parse goals to build the desired stack structure
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                # Ensure the fact has enough parts before accessing indices
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_under[block] = base
                    self.goal_blocks.add(block)
                    # The base block is also part of the goal structure if something is on it
                    self.goal_blocks.add(base)
            elif predicate == "on-table":
                 # Ensure the fact has enough parts before accessing indices
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_on_table.add(block)
                    self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for this heuristic's core count

        # Refine goal_blocks to only include blocks whose position is explicitly specified
        # relative to another block or the table in the goal.
        self.goal_blocks = set(self.goal_under.keys()) | self.goal_on_table


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Extract current block positions from the state
        current_under = {}
        current_on_table = set()
        # current_holding = None # Not needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                 # Ensure the fact has enough parts before accessing indices
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    current_under[block] = base
            elif predicate == "on-table":
                 # Ensure the fact has enough parts before accessing indices
                if len(parts) == 2:
                    block = parts[1]
                    current_on_table.add(block)
            # elif predicate == "holding":
            #     # Ensure the fact has enough parts before accessing indices
            #     if len(parts) == 2:
            #         current_holding = parts[1] # Not used

        # Compute the set of blocks that are in their correct goal stack position
        # relative to the block below them, starting from the table.
        in_goal_stack_position = set()

        # Step 4: Add blocks that are goal_on_table and currently on the table
        for block in self.goal_on_table:
            if block in current_on_table:
                in_goal_stack_position.add(block)

        # Step 5: Iteratively add blocks that are correctly stacked on blocks already in the set
        goal_on_top = {base: block for block, base in self.goal_under.items()}
        
        newly_added = True
        while newly_added:
            newly_added = False
            # Iterate through blocks that should be on top of blocks already in_goal_stack_position
            # Use a list copy to allow modification of the set during iteration
            for base in list(in_goal_stack_position):
                # Check if there is a block that should be on top of this base according to the goal
                if base in goal_on_top:
                    block_on_top_goal = goal_on_top[base]
                    
                    # Check if this block is currently directly on the correct base
                    if block_on_top_goal in current_under and current_under[block_on_top_goal] == base:
                         # Check if this block is already marked in_goal_stack_position
                         if block_on_top_goal not in in_goal_stack_position:
                            in_goal_stack_position.add(block_on_top_goal)
                            newly_added = True

        # Step 6: The heuristic value is the number of goal_blocks not in the correctly built stacks
        heuristic_value = len(self.goal_blocks - in_goal_stack_position)

        # Note: This heuristic is 0 if and only if all blocks that are part of the
        # goal stack structure are correctly placed relative to the stack below them,
        # all the way down to the table. This corresponds to the 'on' and 'on-table'
        # goal predicates being satisfied in a bottom-up manner. If the goal includes
        # 'clear' or 'arm-empty', these are not explicitly counted, but achieving
        # the correct stack structure is the primary challenge in Blocksworld.
        # If the state is the goal state, all goal 'on' and 'on-table' facts are true,
        # and thus all goal_blocks will be in_goal_stack_position, resulting in h=0.

        return heuristic_value

