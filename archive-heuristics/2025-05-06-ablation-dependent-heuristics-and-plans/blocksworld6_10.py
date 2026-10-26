from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Placeholder for the base class if not provided in the execution environment
# In a real planning system, this would be provided by the framework.
class Heuristic:
    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of "misplaced" elements in the state.
    It counts:
    1. A penalty if the arm is not empty when the goal requires it.
    2. For each block that is currently held by the arm.
    3. For each block that is part of a goal stack but is on the wrong support.
    4. For each block that is part of a goal stack, is on the correct support,
       but has the wrong block on top (or should be clear but isn't).
    5. For each block that is not part of any goal stack but is not on the
       table and clear (i.e., it is cluttering the workspace).

    The heuristic is non-admissible and designed to guide a greedy best-first
    search by prioritizing states that appear closer to the goal configuration
    based on these counts.

    # Assumptions
    - Standard Blocksworld domain rules apply.
    - Goal states typically involve specific stack configurations on the table
      and an empty arm. A goal predicate `(arm-empty)` is assumed unless
      `(holding ?x)` is specified in the goal (which is non-standard for goals).
    - All objects (blocks) are present in the initial state facts.

    # Heuristic Initialization
    - Collect all unique block objects from the initial state and goal facts.
    - Parse the goal facts to determine the target support (`on` or `on-table`)
      and the target block that should be directly on top (`on` or `clear`)
      for each block that is part of the goal configuration.
    - Identify the set of blocks that are part of the goal configuration.
    - Check if `(arm-empty)` is required in the goal. Static facts are noted
      but not used as they are typically empty in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic count to 0.
    2. Check if the goal requires the arm to be empty and if it is not empty
       in the current state. If so, add 1 to the count.
    3. Parse the current state to determine the current position (on a block,
       on the table, or held by the arm) and the block currently directly on
       top for every block.
    4. For each block in the problem:
       a. Determine the block's target state: its required support (another block
          or the table) and what should be directly on top of it (another block
          or nothing/clear). Blocks not in goal stacks are targeted to be on
          the table and clear.
       b. Get the block's current position and current top block.
       c. If the block is currently held by the arm, add 1 to the count (it needs
          to be placed somewhere).
       d. Else if the block is part of the goal configuration:
          - If its current position (support) is different from its target support,
            add 1 to the count.
          - Elif its current top block is different from its target top block,
            add 1 to the count.
       e. Else (the block is not part of the goal configuration):
          - If it is not currently on the table and clear, add 1 to the count
            (it is considered clutter that needs to be moved out of the way).
    5. Return the total count.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and objects.
        """
        super().__init__(task)

        # Collect all unique objects (blocks) from initial state and goal
        self.all_blocks = set()
        # Collect from initial state
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if parts:
                 # Objects are arguments after the predicate
                 self.all_blocks.update(parts[1:])
        # Collect from goal state
        for goal in self.goals:
             parts = get_parts(goal)
             if parts:
                 self.all_blocks.update(parts[1:])

        # Parse goal facts to build target configuration
        self.goal_support = {} # block -> support (block or 'table')
        self.goal_top = {}     # block -> block_on_top (block or 'clear')
        self.goal_blocks = set() # blocks explicitly mentioned in goal on/on-table

        # Check if arm-empty is a goal
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_top[support] = block # support has block on top
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'arm-empty' and len(parts) == 1:
                 self.goal_arm_empty = True
            # Note: (clear ?x) and (holding ?x) as goals are handled implicitly
            # or are non-standard for typical blocksworld goals.

        # For blocks in goal_blocks that are supports but nothing is on them in goal,
        # their target_top is 'clear'. This is handled by .get(B, 'clear') later.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        current_pos = {}     # block -> position (block, 'table', or 'arm')
        current_top = {}     # block -> block_on_top (block or 'clear')
        arm_is_empty = False

        # Initialize current_top for all blocks to 'clear' by default
        for block in self.all_blocks:
             current_top[block] = 'clear'

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block, support = parts[1], parts[2]
                current_pos[block] = support
                current_top[support] = block # support has block on top
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                block = parts[1]
                current_pos[block] = 'arm'
            elif predicate == 'arm-empty' and len(parts) == 1:
                 arm_is_empty = True
            # Note: (clear ?x) facts are redundant if we build current_top from 'on' facts

        total_cost = 0

        # 1. Penalty for arm not being empty when required
        if self.goal_arm_empty and not arm_is_empty:
             total_cost += 1

        # 2. Check each block
        for block in self.all_blocks:
            # Determine target state for this block
            target_support = self.goal_support.get(block, 'table') # Default to on-table if not in goal stacks
            target_block_on_top = self.goal_top.get(block, 'clear') # Default to clear if not a support in goal stacks

            # Determine current state for this block
            current_pos_val = current_pos.get(block) # Will be None if block isn't in state facts (shouldn't happen in valid PDDL)
            current_block_on_top_val = current_top.get(block, 'clear') # Default to clear if nothing is on it

            if current_pos_val == 'arm':
                # Block is held. It needs to be placed. Count as 1 action needed for this block.
                total_cost += 1
            elif block in self.goal_blocks:
                # Block is part of a goal stack
                if current_pos_val != target_support:
                    total_cost += 1
                elif current_block_on_top_val != target_block_on_top:
                    total_cost += 1
            else:
                # Block is not part of a goal stack. It should be on the table and clear.
                if current_pos_val != 'table' or current_block_on_top_val != 'clear':
                     total_cost += 1 # It's clutter

        return total_cost
