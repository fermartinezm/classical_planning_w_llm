from heuristics.heuristic_base import Heuristic
# No fnmatch needed for this specific heuristic logic
# No frozenset needed explicitly for calculation, state is already frozenset
# Need set and dict

def get_parts(fact):
    """Helper function to parse PDDL fact strings."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    # Returns a list of strings ['predicate', 'arg1', 'arg2']
    # Handles potential malformed facts by returning an empty list.
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
         return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    Estimates the number of blocks that are not currently part of a correctly
    built goal stack prefix starting from the table. A block is considered
    part of a correctly built goal stack prefix if it is in its correct
    goal position relative to its goal parent, and its goal parent is also
    part of a correctly built goal stack prefix (recursively down to the table).
    The heuristic value is the total number of blocks that are part of the
    goal configuration minus the count of blocks that are part of a correctly
    built goal stack prefix. This counts the number of blocks from the goal
    configuration that are currently "misplaced" relative to the desired
    bottom-up stack structure.

    Assumptions:
    - The goal state consists primarily of 'on' and 'on-table' predicates
      defining the desired stack configurations. 'clear' goals are typically
      implied by the stack structure and are not explicitly used in the
      heuristic calculation logic, although they are part of the goal check.
    - All blocks involved in the goal configuration are assumed to be present
      in the initial state and subsequent states.
    - The input state is a frozenset of PDDL fact strings.
    - The task object provides goals as a frozenset of PDDL fact strings
      and static facts (Blocksworld has no relevant static facts).
    - PDDL fact strings representing state predicates are well-formed
      (e.g., '(predicate arg1 arg2)').

    Heuristic Initialization:
    The constructor parses the goal predicates ('on' and 'on-table') from
    `task.goals` to build the desired goal configuration. It creates a
    dictionary `self.goal_config` mapping each block mentioned in an 'on'
    or 'on-table' goal predicate to its intended parent block (a string
    representing another block) or the string 'table' if the block should
    be on the table in the goal state. It also collects the set of all blocks
    involved in these goal predicates into `self.goal_blocks`. 'clear' goals
    and other predicates are ignored during initialization.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state `node.state` is the goal state `self.goals`.
       If `self.goals` is a subset of `state`, the goal is reached, and the
       heuristic is 0.
    2. Parse the current state to determine the current position of each block.
       Iterate through the facts in `state`. For each `(on block parent)` fact,
       record that `block` is currently on `parent`. For each `(on-table block)`
       fact, record that `block` is currently on the table. Collect all blocks
       mentioned in 'on', 'on-table', 'clear', or 'holding' facts into a set
       `all_blocks_in_state`. Create a dictionary `current_config` mapping each
       block in `all_blocks_in_state` to its current parent block or 'table'.
       A block is on the table if it is not recorded as being 'on' any other block.
    3. Initialize an empty set `blocks_in_goal_stack`. This set will store
       blocks that are part of a correctly built goal stack prefix in the
       current state, according to the goal configuration.
    4. Iteratively populate `blocks_in_goal_stack`: Repeat the following loop
       until a full pass over the blocks adds no new blocks to the set,
       indicating convergence:
       a. Set a `changed` flag to False at the beginning of the pass.
       b. For each block `b` that is part of the goal configuration (i.e., `b`
          is in `self.goal_blocks`):
       c. If `b` is already in `blocks_in_goal_stack`, skip it as it's already
          identified as correctly placed within a goal stack prefix.
       d. Get the current parent `current_parent` of `b` from `current_config`.
          (If `b` is in `self.goal_blocks` but not in `all_blocks_in_state`,
           `current_config.get(b)` will return `None`. This case implies an
           invalid state representation in Blocksworld).
       e. Get the goal parent `goal_parent` of `b` from `self.goal_config`.
          (This will not be `None` because we iterate over `self.goal_blocks`).
       f. If `current_parent` is the same as `goal_parent`: This block `b` is
          in the correct position relative to its immediate goal parent.
          i. If `goal_parent` is the string 'table': This block `b` is correctly
             placed on the table according to the goal. Add `b` to
             `blocks_in_goal_stack`. Set `changed` to True.
          ii. If `goal_parent` is a block: This block `b` is correctly placed
              on its goal parent block. Check if the goal parent block (`goal_parent`)
              is already in `blocks_in_goal_stack`. If it is, it means the stack
              below `b` is correctly built up to `goal_parent`. Add `b` to
              `blocks_in_goal_stack`. Set `changed` to True.
    5. After the iterative process converges, `blocks_in_goal_stack` contains
       all blocks from the goal configuration that are currently part of a
       correctly built goal stack prefix.
    6. The heuristic value is calculated as the total number of blocks in the
       goal configuration (`len(self.goal_blocks)`) minus the number of blocks
       found to be in a correctly built goal stack prefix (`len(blocks_in_goal_stack)`).
       This value represents the number of goal blocks that are not yet fixed
       in their correct relative position within a goal stack built from the
       table upwards.

    """
    def __init__(self, task):
        self.goals = task.goals

        # Parse goal configuration: block -> goal_parent (or 'table')
        self.goal_config = {}
        # Set of all blocks involved in the goal 'on' or 'on-table' predicates
        self.goal_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                self.goal_config[block] = parent
                self.goal_blocks.add(block)
                self.goal_blocks.add(parent)
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_config[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear' goals and other predicates for building goal_config

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Parse current configuration: block -> current_parent (or 'table')
        current_config = {}
        on_map = {} # parent -> block_on_top
        all_blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'on' and len(parts) == 3:
                block, parent = parts[1], parts[2]
                on_map[parent] = block
                all_blocks_in_state.add(block)
                all_blocks_in_state.add(parent)
            elif parts[0] == 'on-table' and len(parts) == 2:
                block = parts[1]
                all_blocks_in_state.add(block)
            elif parts[0] in ['clear', 'holding'] and len(parts) == 2:
                 all_blocks_in_state.add(parts[1])
            elif parts[0] == 'arm-empty':
                 pass # No block involved

        # Determine current parent for each block found in the state
        # A block is on the table if it's not a value in the on_map
        blocks_that_are_on_something = set(on_map.values())
        for block in all_blocks_in_state:
             if block in blocks_that_are_on_something:
                 # Find the parent p such that on_map[p] == block
                 # Assumes valid blocksworld state where each block has at most one parent
                 # This next() call assumes the block is found as a child, which should be true
                 # if it's in blocks_that_are_on_something.
                 parent = next(p for p, child in on_map.items() if child == block)
                 current_config[block] = parent
             else:
                 # If not on something, it must be on the table
                 current_config[block] = 'table'


        # Identify blocks that are part of a correctly built goal stack prefix
        blocks_in_goal_stack = set()
        changed = True
        while changed:
            changed = False
            # Iterate over blocks that are part of the goal configuration
            for block in self.goal_blocks:
                if block in blocks_in_goal_stack:
                    continue # Already correctly placed

                # Get current and goal parents
                # If a block in goal_blocks is not in current_config, it's an invalid state
                # based on Blocksworld assumptions (blocks don't disappear).
                # We use .get() for safety, but expect the block to be present.
                current_parent = current_config.get(block)
                goal_parent = self.goal_config.get(block)

                # This check should technically not be needed if goal_blocks is correctly built
                # from goal_config, and we iterate over goal_blocks.
                if goal_parent is None:
                     continue

                # Check if the block is in its correct position relative to its goal parent
                if current_parent == goal_parent:
                    # Check if the goal parent is correctly placed (or is the table)
                    if goal_parent == 'table':
                        # Block is on the table and should be on the table
                        if block not in blocks_in_goal_stack:
                            blocks_in_goal_stack.add(block)
                            changed = True
                    elif goal_parent in blocks_in_goal_stack:
                        # Block is on the correct parent, and the parent is correctly placed
                        if block not in blocks_in_goal_stack:
                            blocks_in_goal_stack.add(block)
                            changed = True

        # The heuristic is the number of blocks in the goal configuration
        # that are NOT part of a correctly built goal stack prefix.
        heuristic_value = len(self.goal_blocks) - len(blocks_in_goal_stack)

        # The heuristic value is guaranteed to be non-negative because
        # blocks_in_goal_stack is a subset of goal_blocks (we only iterate
        # over goal_blocks when adding to blocks_in_goal_stack).
        # It is 0 if and only if all goal_blocks are in blocks_in_goal_stack,
        # which happens precisely when the state is the goal state.
        # It is finite as the calculation is finite.

        return heuristic_value
