# from heuristics.heuristic_base import Heuristic # Assuming this is provided externally

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace after removing parentheses
    return fact[1:-1].strip().split()

# Define the Heuristic base class if it's not truly external and needed for syntax check
# In a real scenario, this would be in heuristics/heuristic_base.py
# class Heuristic:
#     """Base class for heuristics."""
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

class blocksworldHeuristic: # Removed inheritance for standalone code output as per instructions
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting two types of unsatisfied conditions:
    1. Blocks that are not positioned directly on the correct block or the table
       as specified by the goal state.
    2. Blocks that are required to be clear in the goal state but are not clear
       in the current state.

    # Assumptions
    - The heuristic assumes standard Blocksworld actions (pick-up, put-down,
      stack, unstack) and predicates (on, on-table, clear, holding, arm-empty).
    - All blocks mentioned in the initial or goal state are the relevant objects.
    - The goal state defines the desired configuration using 'on' and 'on-table'
      predicates, and optionally 'clear' for the top blocks.

    # Heuristic Initialization
    - Parses the goal state to determine the desired support for each block
      (which block it should be on, or if it should be on the table). This
      information is stored in `self.goal_below`.
    - Identifies which blocks must be clear in the goal state, storing them
      in `self.goal_clear`.
    - Collects all unique block objects present in the initial state or goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Determine the current configuration of blocks:
       - Identify which block is currently held by the arm (if any).
       - For each block that is not held, determine what it is currently resting on
         (another block or the table). Store this in `current_position_below`.
       - Determine which blocks are currently clear (i.e., nothing is on top of them
         and they are not held by the arm). Store this in `current_clear`.
    2. Initialize the heuristic value `h` to 0.
    3. Count misplaced supports:
       - Iterate through all blocks that have a defined position below them in the goal
         state (i.e., are keys in `self.goal_below`).
       - For each such block `B`, compare its goal support (`self.goal_below[B]`)
         with its current support.
       - If the block `B` is currently held, its support is effectively the arm. Since
         the goal support is never the arm, a held block with a goal support is
         considered misplaced. Increment `h`.
       - If the block `B` is not held, check its current support in `current_position_below`.
         If the current support does not match the goal support, increment `h`.
    4. Count unsatisfied clear goals:
       - Iterate through all blocks that are required to be clear in the goal state
         (i.e., are in `self.goal_clear`).
       - For each such block `B`, check if it is in the `current_clear` set.
       - If `B` is not in `current_clear` (meaning something is on it or it is held),
         increment `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying all blocks.
        """
        self.goals = task.goals

        # 1. Parse goal facts to build goal structure and clear conditions
        self.goal_below = {}  # Maps block -> block_below_it or 'table'
        self.goal_clear = set() # Set of blocks that must be clear in the goal
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                self.goal_below[block_on] = block_below
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                self.goal_below[block_on_table] = 'table'
            elif predicate == 'clear' and len(parts) == 2:
                block_to_be_clear = parts[1]
                self.goal_clear.add(block_to_be_clear)
            # Ignore other goal predicates like arm-empty if they exist

        # 2. Collect all unique block objects from initial state and goal state
        self.all_blocks = set()
        # Collect from initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments that are not 'table'
             for part in parts[1:]:
                 if part != 'table':
                     self.all_blocks.add(part)
        # Collect from goal state
        for goal in self.goals:
             parts = get_parts(goal)
             # Add all arguments that are not 'table'
             for part in parts[1:]:
                 if part != 'table':
                     self.all_blocks.add(part)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1a. Determine current positions and held block
        current_position_below = {} # Maps block -> block_below_it or 'table'
        held_block = None
        # Keep track of blocks that have something on them
        blocks_with_something_on_them = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Skip empty facts
                continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                block_on = parts[1]
                block_below = parts[2]
                current_position_below[block_on] = block_below
                blocks_with_something_on_them.add(block_below)
            elif predicate == 'on-table' and len(parts) == 2:
                block_on_table = parts[1]
                current_position_below[block_on_table] = 'table'
            elif predicate == 'holding' and len(parts) == 2:
                held_block = parts[1]
            # Ignore arm-empty and clear facts for this step

        # 1b. Determine which blocks are currently clear
        # A block is clear if nothing is on it AND it's not held by the arm
        current_clear = {
            block for block in self.all_blocks
            if block not in blocks_with_something_on_them and block != held_block
        }

        # 2. Initialize heuristic value
        h = 0

        # 3. Count misplaced supports
        for block in self.all_blocks:
            if block in self.goal_below: # Only consider blocks that have a defined goal position below them
                goal_pos = self.goal_below[block]

                if held_block == block:
                    # If the block is held, its current "support" is the arm.
                    # Since goal_below is never 'arm', a held block with a goal_below is misplaced.
                    h += 1
                elif block in current_position_below:
                    current_pos = current_position_below[block]
                    if current_pos != goal_pos:
                        h += 1
                # Else: Block is not held and not in current_position_below. This implies
                # it's not on anything and not on the table. This shouldn't happen
                # in a valid Blocksworld state for a block that exists in the problem.
                # We assume valid states, so this case is implicitly handled or indicates an error.

        # 4. Count unsatisfied clear goals
        for block in self.goal_clear:
            if block not in current_clear:
                h += 1

        # 5. Return the total heuristic value
        return h
