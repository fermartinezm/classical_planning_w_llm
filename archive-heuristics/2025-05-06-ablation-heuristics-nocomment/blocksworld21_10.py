from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse a fact string
def get_parts(fact):
    """Splits a fact string into its predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    The heuristic estimates the cost to reach the goal by counting the number
    of goal blocks that are not currently in their correct position relative
    to the goal stack below them, down to the table, and are not correctly
    cleared if they are the top of a goal stack. Each such 'misplaced' goal
    block is estimated to require at least 2 actions (one to pick/unstack,
    one to put/stack) to move it towards its goal position, after any blocks
    on top of it are cleared. The heuristic value is twice the count of these
    misplaced goal blocks.

    Assumptions:
    - The goal state defines one or more well-formed stacks of blocks on the table.
    - The heuristic focuses on achieving the correct 'on' and 'on-table'
      relationships specified in the goal. Explicit '(clear ?x)' goals are
      considered for blocks that are the designated top of a goal stack.
    - The cost of clearing blocks that are wrongly placed on top of a correctly
      positioned block is implicitly accounted for by the factor of 2 per
      misplaced block, and by the fact that blocks above a misplaced block
      will also be considered misplaced if they are goal blocks (as they won't
      be in the 'correctly_placed' set).

    Heuristic Initialization:
    The constructor pre-processes the goal state to build the desired stack
    structure (`goal_structure`) and identify which blocks should be clear
    in the goal state (`goal_clear_blocks`). It also collects all unique
    block names involved in the problem from the initial state and goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine the current position of each block
       (what it is currently on, or if it's on the table or held). Store this
       in `current_structure`. Also identify which blocks are currently clear.
    2. Identify the set of blocks that are 'correctly placed' in the current
       state relative to the goal stack structure. A block `x` is correctly
       placed if:
       a) It is a goal block (part of the desired final configuration).
       b) Its current position matches its goal position (e.g., if it should
          be on block `y`, it is currently on `y`; if it should be on the table,
          it is currently on the table).
       c) If it should be on block `y`, then `y` must also be correctly placed.
       d) If `x` is supposed to be the top block of a goal stack (i.e., it
          should be clear in the goal), it must be currently clear.
       This set (`correctly_placed`) is computed using a bottom-up approach,
       starting from blocks that should be on the table in the goal and
       recursively adding blocks that are correctly placed on top of
       already correctly placed blocks, ensuring the clear condition for
       stack tops.
    3. Count the number of goal blocks that are *not* in the `correctly_placed` set.
       These are the blocks that are part of the desired final configuration
       but are currently in the wrong place relative to the goal structure
       below them, or are not clear when they should be.
    4. The heuristic value is twice this count. This factor of 2 represents
       the minimum two actions (pickup/unstack and putdown/stack) required
       to move a block into position, after it has been cleared.
    5. If the state is the goal state, the heuristic value is 0. This is
       implicitly handled because all goal blocks will be in `correctly_placed`
       and the count will be 0.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Data structures derived from the goal
        self.goal_structure = {} # block -> block_below (or None for on-table)
        self.goal_clear_blocks = set() # blocks that should be clear in the goal
        self.all_blocks = set() # all block objects in the problem

        # Parse goals to build goal_structure and identify blocks that are on top of others in goal
        blocks_that_are_on_top_in_goal = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                self.goal_structure[x] = y
                blocks_that_are_on_top_in_goal.add(x)
                self.all_blocks.add(x)
                self.all_blocks.add(y)
            elif parts[0] == 'on-table':
                x = parts[1]
                self.goal_structure[x] = None # Use None for on-table
                self.all_blocks.add(x)
            elif parts[0] == 'clear':
                 x = parts[1]
                 # Add to goal_clear_blocks explicitly mentioned clear goals
                 self.goal_clear_blocks.add(x)
                 self.all_blocks.add(x)

        # Identify all blocks that are part of the goal structure (keys or values in goal_structure)
        all_goal_blocks_set = set(self.goal_structure.keys()) | set(self.goal_structure.values())
        all_goal_blocks_set.discard(None) # Remove None which represents the table

        # Add blocks that are goal blocks but nothing is specified to be on top of them in the goal structure
        for block in all_goal_blocks_set:
            if block not in blocks_that_are_on_top_in_goal:
                self.goal_clear_blocks.add(block)

        # Parse initial state to find any blocks not mentioned in goals
        # This ensures we have all block names even if they only appear in the initial state
        for fact in self.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1: # Avoid 'arm-empty'
                 for part in parts[1:]:
                     self.all_blocks.add(part)

    def __call__(self, node):
        state = node.state

        # Data structures derived from the current state
        current_structure = {} # block -> block_below (or None for on-table or 'holding')
        current_clear = set() # blocks that are currently clear

        # Parse state to build current_structure and current_clear
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                x, y = parts[1], parts[2]
                current_structure[x] = y
            elif parts[0] == 'on-table':
                x = parts[1]
                current_structure[x] = None # Use None for on-table
            elif parts[0] == 'holding':
                x = parts[1]
                current_structure[x] = 'holding' # Special value
            elif parts[0] == 'clear':
                x = parts[1]
                current_clear.add(x)

        # Compute the set of blocks that are correctly placed in the goal tower
        correctly_placed = set()
        q = [] # Use list as a queue

        # Start with blocks that should be on the table in the goal
        for block in self.all_blocks:
            if block in self.goal_structure and self.goal_structure[block] is None:
                 q.append(block)

        processed = set() # To avoid cycles or redundant processing

        while q:
            x = q.pop(0) # Dequeue
            if x in processed:
                continue
            processed.add(x)

            # Check if x is a goal block and in its correct position relative to below
            is_x_correct = False
            if x in self.goal_structure: # x is a goal block
                goal_pos_below_x = self.goal_structure[x]
                current_pos_below_x = current_structure.get(x) # Get current position below x (None if on-table or held)

                position_matches = False
                if goal_pos_below_x is None: # Goal: x on table
                    if current_pos_below_x is None: # Current: x on table
                        position_matches = True
                elif current_pos_below_x == goal_pos_below_x: # Goal: x on y, Current: x on y
                    # Check if y is correctly placed (must be in correctly_placed set)
                    # If goal_pos_below_x is None, this condition is skipped.
                    if goal_pos_below_x is None or goal_pos_below_x in correctly_placed:
                         position_matches = True

                if position_matches:
                    # Check if x should be clear and is clear, OR if x should NOT be clear
                    should_be_clear = (x in self.goal_clear_blocks)

                    if should_be_clear:
                         if x in current_clear:
                              is_x_correct = True
                    else: # If x should NOT be clear (i.e. something should be on it in goal)
                         is_x_correct = True # Its position relative to below is correct, that's enough for this step

            if is_x_correct:
                correctly_placed.add(x)
                # Add blocks that should be on x in the goal to the queue
                for z in self.all_blocks:
                    if z in self.goal_structure and self.goal_structure[z] == x:
                        q.append(z)

        # Count the number of goal blocks that are not correctly placed
        misplaced_count = 0
        # Identify all blocks that are part of the goal structure (keys or values in goal_structure)
        all_goal_blocks_set = set(self.goal_structure.keys()) | set(self.goal_structure.values())
        all_goal_blocks_set.discard(None) # Remove None which represents the table

        for block in all_goal_blocks_set:
            if block not in correctly_placed:
                 misplaced_count += 1

        # Heuristic value is twice the number of misplaced goal blocks
        # Each misplaced block needs at least 2 actions (pickup/unstack + putdown/stack)
        # after it's cleared.
        return 2 * misplaced_count
