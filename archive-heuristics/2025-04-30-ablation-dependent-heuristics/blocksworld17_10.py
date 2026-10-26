import sys

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and arguments.
    e.g., '(on b1 b2)' -> ('on', ('b1', 'b2'))
    e.g., '(arm-empty)' -> ('arm-empty', ())
    """
    # Remove surrounding parentheses and split by space
    parts = fact_string.strip('()').split()
    predicate = parts[0]
    args = tuple(parts[1:])
    return predicate, args

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the distance to the goal state by counting
    the number of blocks that are not in their correct goal position
    relative to their immediate support (table or another block), plus
    the number of blocks that are blocking a correctly supported block
    but are not the block that should be there according to the goal.
    It also adds a penalty if the arm is holding a block, representing
    the need for at least one action to release it.

    Assumptions:
    - Standard Blocksworld domain with 'on', 'on-table', 'clear',
      'holding', 'arm-empty' predicates and 'pickup', 'putdown',
      'stack', 'unstack' actions.
    - The heuristic is designed for greedy best-first search and is
      NOT admissible. Its goal is to guide the search efficiently by
      providing a good estimate of the remaining work.
    - The heuristic value is 0 if and only if the state is a goal state.
    - The heuristic value is finite for any reachable state.

    Heuristic Initialization:
    In the __init__ method, the heuristic precomputes information from
    the task's goal state and initial state:
    - It builds a `goal_support` map indicating for each block specified
      in an 'on' or 'on-table' goal fact, what its required immediate
      support is ('table' or another block).
    - It identifies the set of blocks whose position is explicitly
      specified in the goal ('goal_position_specified_blocks').
    - It identifies all unique objects (blocks) present in the initial
      state or goal state ('all_objects').

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state:
    1.  Parse the current state facts to build two temporary maps:
        - `state_support`: For each block, what its immediate support
          is in the current state ('table', another block, or 'holding').
        - `state_block_on_top`: For each block, which block (if any)
          is directly on top of it in the current state.
        It also checks if the arm is currently holding a block.
    2.  Initialize the heuristic value `h` to 0.
    3.  Part 1 (Misplaced Blocks): Iterate through each block whose
        position is specified in the goal (`goal_position_specified_blocks`).
        If the block's immediate support in the current state (`state_support`)
        is different from its required support in the goal (`goal_support`),
        increment `h`. This counts blocks that are simply in the wrong place
        relative to their base.
    4.  Part 2 (Blocking Blocks): Iterate through all objects (`all_objects`)
        that could potentially be supporting other blocks. For each such block `B`:
        - Check if there is a block `C` directly on top of `B` in the current state.
        - If yes, check if `B` is currently in its correct position relative
          to its goal support (i.e., `B` is in `goal_position_specified_blocks`
          and `state_support[B]` matches `goal_support[B]`).
        - If `B` is correctly supported AND `C` is NOT the block that should
          be directly on top of `B` according to the goal (`goal_support[C]`
          is not `B`), then `C` is blocking a correctly placed block. Increment `h`.
    5.  Arm Penalty: If the arm is currently holding a block, increment `h` by 1.
        This accounts for the necessary action (putdown or stack) to release
        the block and free the arm.
    6.  Return the final value of `h`.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by precomputing goal information.

        @param task: The planning task object.
        """
        self.goal_support = {}
        self.goal_position_specified_blocks = set() # Blocks whose position is specified by on/on-table in goal

        # Parse goal facts to build goal_support map and identify goal blocks
        for fact_string in task.goals:
            predicate, args = parse_fact(fact_string)
            if predicate == 'on':
                block, support = args
                self.goal_support[block] = support
                self.goal_position_specified_blocks.add(block)
            elif predicate == 'on-table':
                block = args[0]
                self.goal_support[block] = 'table'
                self.goal_position_specified_blocks.add(block)

        # Get all objects from initial state and goal to build state maps efficiently
        self.all_objects = set()
        for fact_string in task.initial_state:
             predicate, args = parse_fact(fact_string)
             for arg in args:
                 self.all_objects.add(arg)
        for fact_string in task.goals:
             predicate, args = parse_fact(fact_string)
             for arg in args:
                 self.all_objects.add(arg)
        # Remove 'arm-empty' if it was added as an object
        self.all_objects.discard('arm-empty')


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of fact strings).
        @return: The heuristic value (integer).
        """
        # Build state information maps
        state_support = {}
        state_block_on_top = {obj: None for obj in self.all_objects}
        is_holding = None

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'on':
                block, support = args
                state_support[block] = support
                state_block_on_top[support] = block
            elif predicate == 'on-table':
                block = args[0]
                state_support[block] = 'table'
            elif predicate == 'holding':
                is_holding = args[0]
                state_support[is_holding] = 'holding' # Represent holding state

        h = 0

        # Part 1: Count blocks not in goal position relative to goal support
        for block in self.goal_position_specified_blocks:
            goal_supp = self.goal_support[block]
            # Use .get() with None default for blocks that might not be in state_support
            # (e.g., if the state is somehow malformed, or if we iterated over all_objects
            # instead of just blocks found in state_support keys)
            state_supp = state_support.get(block, None)

            if state_supp != goal_supp:
                h += 1

        # Part 2: Count blocks blocking correctly supported blocks
        # Iterate through all blocks that could potentially be supports (all_objects)
        for block_below in self.all_objects:
            block_on_top_in_state = state_block_on_top.get(block_below, None)

            if block_on_top_in_state is not None: # There is a block_above on top of block_below in the state
                block_above = block_on_top_in_state

                # Check if block_below is correctly supported according to the goal structure
                # block_below is correctly supported if it's in goal_position_specified_blocks
                # AND its state_support matches its goal_support.
                is_block_below_correctly_supported_in_state = \
                    (block_below in self.goal_position_specified_blocks and
                     state_support.get(block_below, None) == self.goal_support.get(block_below, None)) # Use .get for goal_support too

                if is_block_below_correctly_supported_in_state:
                    # block_below is correctly supported according to the goal.
                    # Is block_above the block that should be on block_below according to the goal?
                    # This means goal_support[block_above] should be block_below.
                    # Check if block_above is in goal_support and its goal_support is block_below.
                    if self.goal_support.get(block_above, None) != block_below:
                         # block_above is not the block required on top of block_below by the goal. block_above is blocking.
                         h += 1

        # Part 3: Arm penalty
        # Add 1 if the arm is holding a block, as this requires an action to resolve.
        if is_holding is not None:
            h += 1

        return h

